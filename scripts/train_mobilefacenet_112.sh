#!/bin/sh
if [ -f "pretrained/ir152_dim64/best.pth.tar" ]; then
    echo "Skipping the dimensionality reduction command as pretrained/ir152_dim64/best.pth.tar already exists."
else
    python train.py --epochs 4 --optimizer Adam --lr 0.001 --scaf-lr 1e-2 --scaf-scale 32 --copy-output-folder pretrained/ir152_dim64 --wd 5e-4 --deterministic --workers 8 --qat-policy None  --model ir_152 --dr 64 --backbone-checkpoint pretrained/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth --use-bias --dataset VGGFace2_FaceID_dr --batch-size 64 --device MAX78000 --validation-split 0 --print-freq 250 "$@" || exit 1
fi
python train.py --epochs 35 --optimizer Adam --lr 0.001 --compress policies/schedule-mobilefacenet_112.yaml --kd-student-wt 0 --kd-distill-wt 1 --qat-policy policies/qat_policy_mobilefacenet_112.yaml --model ai87netmobilefacenet_112 --kd-teacher ir_152 --kd-resume pretrained/ir152_dim64/best.pth.tar --kd-relationbased --wd 0 --deterministic --workers 8 --use-bias --dataset VGGFace2_FaceID --batch-size 100 --device MAX78002 --validation-split 0 --print-freq 100 "$@"
