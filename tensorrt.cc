
#include "tensorrt.h"

Napi::FunctionReference TensorRT::constructor;

Napi::Object TensorRT::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "TensorRT", {
      InstanceMethod("load", &TensorRT::load),
      InstanceMethod("unload", &TensorRT::unload),
      InstanceMethod("info", &TensorRT::info),
      InstanceMethod("execute", &TensorRT::execute),
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("TensorRT", func);
  return exports;
}

TensorRT::TensorRT(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TensorRT>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  // Napi::Object options;

  // if(info[0].IsObject()) {
  //   options = info[0].As<Napi::Object>();
  // } else {
  //   options = Napi::Object::New(env);
  // }

  // if(options.HasOwnProperty("cache_path")) {
  //   cache_path = options.Get("cache_path").As<Napi::String>();
  // }

  cache_path = std::string(getenv("HOME")) + "/.cache/libtrt";

  struct stat sb;
  if(stat(cache_path.c_str(), &sb)) {
    mkdir(cache_path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
  }

};

TensorRT::~TensorRT() {
  //
}

Napi::Object TensorRT::NewInstance(Napi::Env env, Napi::Value arg) {
  Napi::EscapableHandleScope scope(env);
  Napi::Object obj = constructor.New({ arg });
  return scope.Escape(napi_value(obj)).ToObject();
}

#define CHECK_THREAD(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            throw std::runtime_error("Cuda failure"); \
        }                                                      \
    } while (0)


#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            Napi::Error::New(env, "Cuda failure").ThrowAsJavaScriptException(); \
            return env.Undefined(); \
        }                                                      \
    } while (0)



class Logger : public nvinfer1::ILogger
{
  public:
    Logger()
    {
    }

    //!
    //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
    //! \return The nvinfer1::ILogger associated with this Logger
    //!
    //! TODO Once all samples are updated to use this method to register the logger with TensorRT,
    //! we can eliminate the inheritance of Logger from ILogger
    //!
    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

    //!
    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    //!
    //! Note samples should not be calling this function directly; it will eventually go away once we eliminate the inheritance from
    //! nvinfer1::ILogger
    //!
    void log(Severity severity, const char* msg) override
    {
      if(severity <= Severity::kWARNING) {
        std::cout << severityPrefix(severity) << " " << std::string(msg) << std::endl;
      }
    }

  private:
    static std::string severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }  
};

Logger myLogger;

bool onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory *&trtModelStream) // output buffer for the TensorRT model
{
  // create the builder
  IBuilder *builder = createInferBuilder(myLogger.getTRTLogger());
  assert(builder != nullptr);
  nvinfer1::INetworkDefinition *network = builder->createNetwork();

  auto parser = nvonnxparser::createParser(*network, myLogger.getTRTLogger());

  //Optional - uncomment below lines to view network layer information
  //config->setPrintLayerInfo(true);
  //parser->reportParsingInfo();

  //if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(myLogger.getReportableSeverity()) ) )
  if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
  {
    std::cout << "Failure while parsing ONNX file" << std::endl;
    return false;
  }

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 20);
  // FP32 mode is not permitted with DLA?
  builder->setFp16Mode(true);
  builder->setInt8Mode(false);

  // if (gArgs.runInInt8)
  // {
  //   samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
  // }

#ifdef USEDLACORE
  if (builder->getNbDLACores() > 0)
  {
    builder->allowGPUFallback(true);
    builder->setDefaultDeviceType(DeviceType::kDLA);
    builder->setDLACore(0);
    builder->setStrictTypeConstraints(true);
  }
#endif

  ICudaEngine *engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we can destroy the parser
  parser->destroy();

  // serialize the engine, then close everything down
  trtModelStream = engine->serialize();
  engine->destroy();
  network->destroy();
  builder->destroy();

  return true;
}

const char *get_filename_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if(!dot || dot == filename) return "";
  return dot + 1;
}

Napi::Value TensorRT::load(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  std::string *inputpath = new std::string(info[0].As<Napi::String>());
  std::string *err = new std::string();

  PromiseWorker *wk = new PromiseWorker(env,
  [=]() {

    struct stat sb;
    if(stat(inputpath->c_str(), &sb)) {
      throw std::runtime_error("File not found");
    }
    if((sb.st_mode & S_IFMT) != S_IFREG) {
      throw std::runtime_error("Not a regular File");
    }
    long int inputfile_time = sb.st_mtime;

    bool use_trtfile = false;
    std::string trtfile;

    const char *ext = get_filename_ext(inputpath->c_str());
    if(strcmp(ext, "trt") == 0) {

      use_trtfile = true;
      trtfile = (*inputpath);

    } else {

      long int trtfile_time = 0;

      std::string base = std::string(basename(inputpath->c_str()));

      trtfile = cache_path + "/" + base + ".trt";

      if(!stat(trtfile.c_str(), &sb)) {
        if((sb.st_mode & S_IFMT) == S_IFREG) {
          trtfile_time = sb.st_mtime;
        }
      }

      use_trtfile = (trtfile_time > inputfile_time);

      printf("TensorRT::load: %s \n", (use_trtfile) ? "Using Cache" : "Parsing ONNX");

    }

    //return Napi::String::New(env, trtfile);

    // deserialize the engine
    //IRuntime *runtime = createInferRuntime(myLogger);
    runtime = createInferRuntime(myLogger);
    assert(runtime != nullptr);
#ifdef USEDLACORE
    if (runtime->getNbDLACores() > 0)
    {
      runtime->setDLACore(0);
    }
#endif

    //ICudaEngine *engine{nullptr};

    if (use_trtfile) {
      std::vector<char> trtModelStream;	
      size_t size{ 0 };
      std::ifstream file(trtfile, std::ios::binary);
      if (file.good())
      {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();
      }
      engine = runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size(), nullptr);
      assert(engine != nullptr);
    } else {
      //create a TensorRT model from the onnx model and serialize it to a stream
      IHostMemory *trtModelStream{nullptr};
      onnxToTRTModel(inputpath->c_str(), 1, trtModelStream);
      assert(trtModelStream != nullptr);

      std::ofstream p(trtfile, std::ios::binary);
      p.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
      p.close();

      engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
      assert(engine != nullptr);
      trtModelStream->destroy();
    }

    //IExecutionContext *context = engine->createExecutionContext();
    context = engine->createExecutionContext();
    assert(context != nullptr);

    nbBindings = engine->getNbBindings();
    if(nbBindings > MAXBINDINGS) {
      throw std::runtime_error("nbBindings > MAXBINDINGS");
    }

    //std::cout << "bindings: " << nbBindings << std::endl;
    for (int i = 0; i < nbBindings; ++i)
    {
      //bool isinput = engine->bindingIsInput(i);
      //std::cout << "binding " << i << ":" << engine->getBindingName(i) << ((isinput) ? " is input " : " is output ");
      Dims bufdims = engine->getBindingDimensions(i);
      //std::cout << "[ ";
      int dataCnt = 1;
      for (int j = 0; j < bufdims.nbDims; j++)
      {
        //std::cout << bufdims.d[j] << ((j == bufdims.nbDims - 1) ? " ]" : ", ");
        dataCnt *= bufdims.d[j];
      }
      //std::cout << std::endl;
      bufsize[i] = dataCnt;
      cpubuf[i] = (float *)malloc(dataCnt * sizeof(float));
      for (int j = 0; j < dataCnt; j++)
      {
        cpubuf[i][j] = 0.0;
      }
      CHECK_THREAD(cudaMalloc(&gpubuf[i], dataCnt * sizeof(float)));
    }

  },
  [=](Napi::Env env) -> Napi::Value {

    //cleanup what you allocated
    delete inputpath;
    delete err;

    return env.Undefined();
  });
  wk->Queue();
  return wk->Deferred().Promise();
}

Napi::Value TensorRT::unload(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  for (int i = 0; i < nbBindings; i++)
  {
    free(cpubuf[i]);
    CHECK(cudaFree(gpubuf[i]));
  }

  context->destroy();
  engine->destroy();
  runtime->destroy();

  return env.Undefined();
}

Napi::Value TensorRT::info(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  Napi::Object retval = Napi::Object::New(env);
  Napi::Array inputs = Napi::Array::New(env);
  Napi::Array outputs = Napi::Array::New(env);  
  retval.Set("inputs", inputs);
  retval.Set("outputs", outputs);

  int cnt_inp = 0;
  int cnt_out = 0;

  for (int i = 0; i < nbBindings; ++i)
  {
    bool isinput = engine->bindingIsInput(i);
    std::string name(engine->getBindingName(i));

    Dims bufdims = engine->getBindingDimensions(i);
    Napi::Array arrdim = Napi::Array::New(env);
    for (int j = 0; j < bufdims.nbDims; j++)
    {
      arrdim.Set(j, Napi::Number::New(env, bufdims.d[j]));
    }

    Napi::Object binding = Napi::Object::New(env);
    binding.Set("name", Napi::String::New(env, name));
    binding.Set("dim", arrdim);

    if(isinput) {
      inputs.Set(cnt_inp++, binding);
    } else {
      outputs.Set(cnt_out++, binding);
    }
  }

  return retval;
}


Napi::Value TensorRT::execute(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if(!info[0].IsObject()) {
    Napi::Error::New(env, "No Input").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  Napi::Object inputs = info[0].As<Napi::Object>();

  for (int i = 0; i < nbBindings; ++i)
  {
    bool isinput = engine->bindingIsInput(i);
    if(isinput) {
      std::string name(engine->getBindingName(i));
      Dims bufdims = engine->getBindingDimensions(i);
      if(!inputs.HasOwnProperty(name)) {
        Napi::Error::New(env, "Bad Input").ThrowAsJavaScriptException();
        return env.Undefined();
      }
      Napi::Object input = inputs.Get(name).As<Napi::Object>();
      Napi::Array arrdim = input.Get("dim").As<Napi::Array>();
      for (int j = 0; j < bufdims.nbDims; j++)
      {
        Napi::Value dimval = arrdim.Get(j);
        if(!dimval.IsNumber()) {
          Napi::Error::New(env, "Bad Dimensions").ThrowAsJavaScriptException();
          return env.Undefined();
        }
        int d = dimval.As<Napi::Number>().Int32Value();
        if(bufdims.d[j] != d) {
          Napi::Error::New(env, "Bad Dimensions").ThrowAsJavaScriptException();
          return env.Undefined();
        }
      }

      Napi::TypedArray arrdata = input.Get("data").As<Napi::TypedArray>();
      Napi::Float32Array arrdata_float = arrdata.As<Napi::Float32Array>();
      std::copy(arrdata_float.Data(), arrdata_float.Data() + bufsize[i], cpubuf[i]);
    }
  }

  PromiseWorker *wk = new PromiseWorker(env,
  [=]() {

    cudaStream_t stream;
    CHECK_THREAD(cudaStreamCreate(&stream));

    for (int i = 0; i < nbBindings; i++)
    {
      if(engine->bindingIsInput(i)) {
        CHECK_THREAD(cudaMemcpyAsync(gpubuf[i], cpubuf[i], 1 * bufsize[i] * sizeof(float), cudaMemcpyHostToDevice, stream));
      }
    }

    context->enqueue(1, gpubuf, stream, nullptr);

    for (int i = 0; i < nbBindings; i++)
    {
      if(!engine->bindingIsInput(i)) {
        CHECK_THREAD(cudaMemcpyAsync(cpubuf[i], gpubuf[i], 1 * bufsize[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
      }
    }

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

  },
  [=](Napi::Env env) -> Napi::Value {

    Napi::Object retval = Napi::Object::New(env);

    for (int i = 0; i < nbBindings; ++i)
    {
      bool isoutput = !engine->bindingIsInput(i);
      if(isoutput) {
        std::string name(engine->getBindingName(i));

        Dims bufdims = engine->getBindingDimensions(i);
        Napi::Array arrdim = Napi::Array::New(env);
        for (int j = 0; j < bufdims.nbDims; j++)
        {
          arrdim.Set(j, Napi::Number::New(env, bufdims.d[j]));
        }
        
        Napi::Object binding = Napi::Object::New(env);
        binding.Set("dim", arrdim);
        Napi::Float32Array data = Napi::Float32Array::New(env, bufsize[i]);
        std::copy(cpubuf[i], cpubuf[i] + bufsize[i], data.Data());
        binding.Set("data", data);

        retval.Set(name, binding);

      }
    }

    return retval;

  });
  wk->Queue();
  return wk->Deferred().Promise();
}


// Napi::Value TensorRT::fnAsync(const Napi::CallbackInfo &info)
// {
//   Napi::Env env = info.Env();
//   Napi::Object inputobj = info[0].As<Napi::Object>();
  
//   //Memory allocation of inputs and outputs to persist through the async call
//   Napi::Buffer<unsigned char> jsdata = inputobj.Get("data").As<Napi::Buffer<unsigned char>>();
//   std::vector<unsigned char> *input1 = new std::vector<unsigned char>(jsdata.Data(), jsdata.Data()+jsdata.ByteLength());

//   std::string *output1 = new std::string();

//   PromiseWorker *wk = new PromiseWorker(env,
//   [=]() {

//     //This part is in a separate thread
//     //You can't interact with V8/Napi here
    
//     //(*output1) = blocking_fn(input1->data());

//   },
//   [=](Napi::Env env) -> Napi::Value {

//     Napi::Object retval = Napi::Object::New(env);
//     retval.Set("output1", (*output1));

//     //cleanup what you allocated
//     delete input1;
//     delete output1;

//     return retval;
//   });
//   wk->Queue();
//   return wk->Deferred().Promise();
// }

// Napi::Value TensorRT::fnSync(const Napi::CallbackInfo &info)
// {
//   Napi::Env env = info.Env();
//   Napi::Object inputobj = info[0].As<Napi::Object>();
  
//   Napi::Buffer<unsigned char> jsdata = inputobj.Get("data").As<Napi::Buffer<unsigned char>>();

//   std::string output1;

//   //output1 = blocking_fn(jsdata.Data());

//   Napi::Object retval = Napi::Object::New(env);

//   retval.Set("output1", output1);

//   return retval;
// }