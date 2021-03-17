#!/bin/sh
./train.py --epochs 25000 --optimizer SGD --lr 0.001 --model ai85nasnet_sequential_cifar100 --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 250 --validation-split 0 --qat-policy None --nas --nas-policy nas/nas_policy_cifar100.yaml --use-bias --load-serialized "$@"
