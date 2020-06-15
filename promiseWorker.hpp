#ifndef PROMISEWORKER_H
#define PROMISEWORKER_H

#include <napi.h>

class PromiseWorker : public Napi::AsyncWorker {
  public:
    PromiseWorker(Napi::Env env, 
      std::function<void ()> task, 
      std::function<Napi::Value (Napi::Env env)> resolver)
    : Napi::AsyncWorker(Napi::Function::New(env, [](const Napi::CallbackInfo& info){})), 
      env(env), deferred(Napi::Promise::Deferred::New(env)), task(task), resolver(resolver),
      errstack(Napi::Error::New(env, "")) {}
    ~PromiseWorker() {}

    void Execute() {
      try {
        task();
      } catch (const std::exception& e) {
        SetError(e.what());
      }
    }

    Napi::Promise::Deferred& Deferred() {
      return deferred;
    }

    void OnOK() {
      Napi::EscapableHandleScope scope(env);
      if(resolver) {
        deferred.Resolve(scope.Escape(napi_value(resolver(env))));
      } else {
        deferred.Resolve(env.Undefined());
      }
    }

    void OnError(const Napi::Error& e) {
      Napi::HandleScope scope(env);
      Napi::Object errobj = e.Value().As<Napi::Object>();
      Napi::Object curstack = errstack.Value().As<Napi::Object>();
      std::string stack = curstack.Get("stack").As<Napi::String>();
      std::string newstack = e.Message() + "\n" + stack;
      errobj.Set("stack", Napi::String::New(env, newstack));
      deferred.Reject(errobj);
    }

  private:
    Napi::Env env;
    Napi::Promise::Deferred deferred;
    std::function<void ()> task;
    std::function<Napi::Value (Napi::Env env)> resolver;
    Napi::Error errstack;
};

#endif
