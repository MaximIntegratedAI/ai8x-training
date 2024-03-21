#!/bin/sh
python train.py --model ai85kws20netnas --use-bias --dataset KWS_20_msnoise_mixed --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws20_nas_msnoise_mixed-qat8-q.pth.tar -8 --device MAX78000 "$@"
