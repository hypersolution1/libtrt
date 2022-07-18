
#include "tensorrt.h"

Napi::FunctionReference TensorRT::constructor;

Napi::Object TensorRT::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "TensorRT", {
      InstanceMethod("load", &TensorRT::load),
      InstanceMethod("unload", &TensorRT::unload),
      InstanceMethod("info", &TensorRT::info),
      InstanceMethod("execute", &TensorRT::execute),
      InstanceMethod("yolo", &TensorRT::yolo),
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("TensorRT", func);
  return exports;
}

TensorRT::TensorRT(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TensorRT>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  Napi::Object options;

  if(info[0].IsObject()) {
    options = info[0].As<Napi::Object>();
  } else {
    options = Napi::Object::New(env);
  }

  if(options.HasOwnProperty("cache_path")) {
    cache_path = options.Get("cache_path").As<Napi::String>();
  } else {
    cache_path = std::string(getenv("HOME")) + "/.cache/libtrt";
  }

  if(options.HasOwnProperty("fp16")) {
    mode_fp16 = options.Get("fp16").As<Napi::Boolean>();
  } else {
    mode_fp16 = false;
  }

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
                    IHostMemory *&trtModelStream,  // output buffer for the TensorRT model
                    bool mode_fp16)
{
  // create the builder
  IBuilder *builder = createInferBuilder(myLogger.getTRTLogger());
  assert(builder != nullptr);

#if NV_TENSORRT_MAJOR >= 7
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  assert(config);

  config->setMaxWorkspaceSize(16 * (1 << 20));
  if (mode_fp16)
  {
    config->setFlag(BuilderFlag::kFP16);
  }

#else
  nvinfer1::INetworkDefinition *network = builder->createNetwork();
  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 20);
  // FP32 mode is not permitted with DLA?
  builder->setFp16Mode(mode_fp16);
  builder->setInt8Mode(false);
#endif

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

#if NV_TENSORRT_MAJOR >= 7
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  // profile->setDimensions("input_1", OptProfileSelector::kMIN, Dims4(1,96,96,3));
  // profile->setDimensions("input_1", OptProfileSelector::kOPT, Dims4(1,96,96,3));
  // profile->setDimensions("input_1", OptProfileSelector::kMAX, Dims4(1,96,96,3));
  // config->addOptimizationProfile(profile);

  Dims inputDims;
  bool has_profile = false;
  for(int i = 0; i < network->getNbInputs(); i++) {
    ITensor *inp = network->getInput(0);
    inputDims = inp->getDimensions();
    // printf("Input: %s\n", inp->getName());
    // for(int j = 0; j < inputDims.nbDims; j++) {
    //   printf("Dim: %d\n", inputDims.d[j]);
    // }
    if(inputDims.d[0] == -1) {
      has_profile = true;
      inputDims.d[0] = 1;
      profile->setDimensions(inp->getName(), OptProfileSelector::kMIN, inputDims);
      inputDims.d[0] = maxBatchSize;
      profile->setDimensions(inp->getName(), OptProfileSelector::kOPT, inputDims);
      inputDims.d[0] = maxBatchSize;
      profile->setDimensions(inp->getName(), OptProfileSelector::kMAX, inputDims);
    }

  }

  if(has_profile) {
    config->addOptimizationProfile(profile);
  }
#endif

#if NV_TENSORRT_MAJOR >= 7
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
#else
  ICudaEngine *engine = builder->buildCudaEngine(*network);
#endif
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
      onnxToTRTModel(inputpath->c_str(), 1, trtModelStream, mode_fp16);
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

    maxBatchSize = engine->getMaxBatchSize();

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
      dataCnt *= maxBatchSize;
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
  retval.Set("maxBatchSize", Napi::Number::New(env, maxBatchSize));
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
        std::string err("Bad Input, expecting: " + name);
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
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

      Napi::Array arrbatchData;
      if(input.Get("data").IsArray()) {
        input_is_array = true;
        arrbatchData = input.Get("data").As<Napi::Array>();
      } else {
        input_is_array = false;
        arrbatchData = Napi::Array::New(env);
        //Napi::TypedArray arrdata = input.Get("data").As<Napi::TypedArray>();
        arrbatchData.Set((int)0, input.Get("data"));
      }

      batchSize = arrbatchData.Length();
      if(batchSize > maxBatchSize) {
        Napi::Error::New(env, "Input greater than maximum batch size").ThrowAsJavaScriptException();
        return env.Undefined();
      }
      for(int j = 0; j < batchSize; j++) {
        Napi::TypedArray arrdata = arrbatchData.Get(j).As<Napi::TypedArray>();
        Napi::Float32Array arrdata_float = arrdata.As<Napi::Float32Array>();
        std::copy(arrdata_float.Data(), arrdata_float.Data() + bufsize[i], cpubuf[i] + (j * bufsize[i]));
      }

      // Napi::TypedArray arrdata = input.Get("data").As<Napi::TypedArray>();
      // Napi::Float32Array arrdata_float = arrdata.As<Napi::Float32Array>();
      // std::copy(arrdata_float.Data(), arrdata_float.Data() + bufsize[i], cpubuf[i]);

    }
  }

  PromiseWorker *wk = new PromiseWorker(env,
  [=]() {

    cudaStream_t stream;
    CHECK_THREAD(cudaStreamCreate(&stream));

    for (int i = 0; i < nbBindings; i++)
    {
      if(engine->bindingIsInput(i)) {
        CHECK_THREAD(cudaMemcpyAsync(gpubuf[i], cpubuf[i], maxBatchSize * bufsize[i] * sizeof(float), cudaMemcpyHostToDevice, stream));
      }
    }

    context->enqueue(maxBatchSize, gpubuf, stream, nullptr);

    for (int i = 0; i < nbBindings; i++)
    {
      if(!engine->bindingIsInput(i)) {
        CHECK_THREAD(cudaMemcpyAsync(cpubuf[i], gpubuf[i], maxBatchSize * bufsize[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
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

        Napi::Array arrdata = Napi::Array::New(env);
        for(int j = 0; j < batchSize; j++) {
          Napi::Float32Array data = Napi::Float32Array::New(env, bufsize[i]);
          std::copy(cpubuf[i] + (j * bufsize[i]), cpubuf[i] + ((j+1) * bufsize[i]), data.Data());
          arrdata.Set(j, data);
        }
        if(input_is_array) {
          binding.Set("data", arrdata);
        } else {
          //Napi::Value val = .As<Napi::Value>();
          binding.Set("data", arrdata.Get((int)0));
        }

        // Napi::Float32Array data = Napi::Float32Array::New(env, bufsize[i]);
        // std::copy(cpubuf[i], cpubuf[i] + bufsize[i], data.Data());
        // binding.Set("data", data);

        retval.Set(name, binding);

      }
    }

    return retval;

  });
  wk->Queue();
  return wk->Deferred().Promise();
}

void get_rect(int cols, int rows, float input_size, float bbox[4], int *out) {
    int l, r, t, b;
    float r_w = input_size / (cols * 1.0);
    float r_h = input_size / (rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (input_size - r_w * rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (input_size - r_w * rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (input_size - r_h * cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (input_size - r_h * cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    out[0] = l;
    out[1] = t;
    out[2] = r - l;
    out[3] = b - t;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

#define CONF_THRESH 0.5
#define NMS_THRESH 0.4

Napi::Value TensorRT::yolo(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  Napi::Object inputobj = info[0].As<Napi::Object>();

  if(!inputobj.HasOwnProperty("data") || !inputobj.HasOwnProperty("width") || !inputobj.HasOwnProperty("height")) {
    Napi::Error::New(env, "Missing data/width/height").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  float conf_thresh = CONF_THRESH;
  float nms_thresh = NMS_THRESH;
  float input_size = 640.f;

  if(inputobj.HasOwnProperty("conf_thresh")) {
    conf_thresh = inputobj.Get("conf_thresh").As<Napi::Number>().FloatValue();
  }
  if(inputobj.HasOwnProperty("nms_thresh")) {
    nms_thresh = inputobj.Get("nms_thresh").As<Napi::Number>().FloatValue();
  }
  if(inputobj.HasOwnProperty("input_size")) {
    input_size = inputobj.Get("input_size").As<Napi::Number>().FloatValue();
  }

  Napi::TypedArray arrdata = inputobj.Get("data").As<Napi::TypedArray>();
  Napi::Float32Array jsdata = arrdata.As<Napi::Float32Array>();  

  int width = inputobj.Get("width").As<Napi::Number>().Int32Value();
  int height = inputobj.Get("height").As<Napi::Number>().Int32Value();

  //doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
  std::vector<Yolo::Detection> res;
  nms(res, jsdata.Data(), conf_thresh, nms_thresh);

  Napi::Array retval = Napi::Array::New(env);

  for (size_t j = 0; j < res.size(); j++) {
    int rect[4];
    get_rect(width, height, input_size, res[j].bbox, rect);
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("x", rect[0]);
    obj.Set("y", rect[1]);
    obj.Set("w", rect[2]);
    obj.Set("h", rect[3]);
    obj.Set("id", (int)res[j].class_id);
    obj.Set("conf", res[j].conf);
    retval.Set(j, obj);
  }

  return retval;
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
