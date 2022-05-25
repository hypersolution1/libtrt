# libTRT

This library performs non-blocking inference on tensorRT model.

## Features

- Support TensorRT engine model
- Support ONNX model
- Include yolo post processing

## Usage

```js

var model = libTRT()
await model.load("yolov5x.engine") // File generated for the target GPU from a .wts file (see https://github.com/ultralytics/yolov5/releases)

console.log(model.info()) // Model inspection

var input = {}
input['data'] = { // input name in model
  "dim": [3, 640, 640], // Used as a validation of the input
  "data": arrImgs, // Float32Array, OR Array of Float32Array with length not higher than the maximum batch size of the model
}

var out = await model.execute(input) // Return an object with outputs as keys

var objs = model.yolo({ // This library includes a yolo post processing function
  data: out['prob'].data, // OR .data[i] if batch is used
  width: 640,
  height: 640,
  input_size: 640, // or 1280 depending on your model 
  conf_thresh: 0.5,
  nms_thresh: 0.4,
})

```