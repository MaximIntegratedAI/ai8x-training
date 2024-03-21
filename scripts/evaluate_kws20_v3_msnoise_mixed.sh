#!/bin/sh
python train.py --model ai85kws20netv3 --dataset KWS_20_msnoise_mixed --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws20_v3_msnoise_mixed-qat8-q.pth.tar -8 --device MAX78000 "$@"
