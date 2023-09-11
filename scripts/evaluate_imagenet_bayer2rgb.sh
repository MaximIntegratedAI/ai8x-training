#!/bin/sh
python train.py --model bayer2rgbnet --dataset ImageNet_Bayer --evaluate --device MAX78000 --regression --exp-load-weights-from ../ai8x-synthesis/trained/ai85-b2rgb-qat8-q.pth.tar -8 --use-bias --data /data_ssd --save-sample 10 "$@"
