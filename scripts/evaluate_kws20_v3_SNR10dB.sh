#!/bin/sh
./train.py --model ai85kws20netv3 --dataset MixedKWS20_10dB --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws20_v3-qat8-q.pth.tar -8 --device MAX78000 "$@"
