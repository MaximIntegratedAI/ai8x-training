#!/bin/sh
./train.py --epochs 25000 --optimizer SGD --lr 0.001 --model ai85ofanet_cifar100 --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 250 --validation-split 0 --qat-policy None --ofa --ofa-policy ofa_policy.yaml --use-bias "$@"
