#ifndef TENSORRT_H
#define TENSORRT_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <napi.h>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "promiseWorker.hpp"

//#define USEDLACORE 1

#define MAXBINDINGS 8

using namespace nvinfer1;

class TensorRT : public Napi::ObjectWrap<TensorRT> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object NewInstance(Napi::Env env, Napi::Value arg);
    TensorRT(const Napi::CallbackInfo& info);
    ~TensorRT();

  private:
    static Napi::FunctionReference constructor;
    //Napi::Value fnAsync(const Napi::CallbackInfo& info);
    //Napi::Value fnSync(const Napi::CallbackInfo& info);
    Napi::Value load(const Napi::CallbackInfo& info);
    Napi::Value unload(const Napi::CallbackInfo& info);
    Napi::Value info(const Napi::CallbackInfo& info);
    Napi::Value execute(const Napi::CallbackInfo& info);

    //
    std::string cache_path;
    bool mode_fp16;
    IRuntime *runtime{nullptr};
    ICudaEngine *engine{nullptr};
    IExecutionContext *context{nullptr};

    int nbBindings;
    //Dims bufdims[MAXBINDINGS];
    int bufsize[MAXBINDINGS];
    float *cpubuf[MAXBINDINGS];
    void *gpubuf[MAXBINDINGS];

};

#endif
