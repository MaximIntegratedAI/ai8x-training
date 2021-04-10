#!/bin/sh
./train.py --epochs 15000 --optimizer SGD --lr 0.001 --model ai85nasnet_sequential_kws20 --dataset KWS_20 --device MAX78000 --batch-size 100 --print-freq 250 --validation-split 0 --qat-policy None --nas --nas-policy nas/nas_policy_kws20.yaml --use-bias --load-serialized "$@"
