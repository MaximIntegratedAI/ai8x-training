#!/bin/sh
./run_nas_network_search.py --model_path nas/trained/cifar100_nas_sequential_stg3_lev2.pth.tar --arch ai85nasnet_sequential_cifar100 --dataset CIFAR100 --nas-policy nas/nas_policy_cifar100.yaml --num-out-archs 10 --export-archs --arch-file nas/nas_out_subnets_cifar100.json "$@"
