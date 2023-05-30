#!/bin/sh
python train.py --deterministic --print-freq 50 --model ai87fpndetector --use-bias --dataset PascalVOC_2007_2012_256_320_augmented --device MAX78002 --obj-detection --obj-detection-params parameters/obj_detection_params_pascalvoc.yaml --batch-size 32 --qat-policy policies/qat_policy_pascalvoc.yaml --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --validation-split 0 "$@"

