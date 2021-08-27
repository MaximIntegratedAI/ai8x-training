#!/bin/sh
./train.py --model ai85rpsnet --dataset asl_big --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-asl01-chw.pth.tar -8 --device MAX78000 --use-bias "$@"
