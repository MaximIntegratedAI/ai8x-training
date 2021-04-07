#!/bin/sh
./train.py --print-freq 500 --epochs 20 --optimizer Adam --lr 0.001 --deterministic --compress schedule_kws20.yaml --model ai85kwsfew --dataset KWS_few --confusion --device MAX78000 "$@"
