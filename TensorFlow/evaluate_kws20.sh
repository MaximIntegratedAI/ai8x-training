#!/bin/sh
python evaluate.py --onnx-file export/kws20/saved_model.onnx --dataset kws20
python evaluate.py --onnx-file export/kws20/saved_model_dq.onnx --dataset kws20
