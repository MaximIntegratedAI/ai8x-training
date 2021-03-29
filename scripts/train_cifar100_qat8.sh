#!/bin/sh
./train.py --deterministic --epochs 300 --optimizer Adam --lr 0.001 --compress schedule-cifar-nas.yaml --model ai85nascifarnet --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 100 --validation-split 0 --use-bias --qat-policy qat_policy_late_cifar.yaml "$@"
