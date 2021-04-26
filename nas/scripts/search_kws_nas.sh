#!/bin/sh
./run_nas_network_search.py --model_path nas/trained/kws_nas_sequential_stg3_lev2.pth.tar --arch ai85nasnet_sequential_kws20 --dataset KWS_20 --nas-policy nas/nas_policy_kws20.yaml --num-out-archs 20 --export-archs --arch-file nas/nas_out_subnets_kws20.json "$@"
