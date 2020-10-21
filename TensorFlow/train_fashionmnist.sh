#!/bin/sh
./train.py --epochs 100 --batch_size 256 --optimizer Adam --lr 0.001 --model fashionmnist_model --dataset fashionmnist --save-sample 1234 --save-sample-per-class "$@"
./convert.py --saved-model export/fashionmnist --inputs-as-nchw input_1:0 --opset 10 --output export/fashionmnist/saved_model.onnx
