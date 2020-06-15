{
  "targets": [
    {
      "target_name": "addon",
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "sources": [ "tensorrt.cc", "addon.cc" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "/usr/local/cuda/include",
      ],
      "defines": [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ],
      "ldflags": [ ], 
      "link_settings": {
        "libraries": [
          "-lcudart",
          "-lnvinfer",
          "-lnvonnxparser",
        ],
        "library_dirs": [
          "/usr/local/cuda/lib64",
        ]
      }      
    }
  ]
}
