#!/bin/sh
./train.py --model ai85nascifarnet --dataset CIFAR100 --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar100-qat8-q.pth.tar -8 --use-bias "$@"
