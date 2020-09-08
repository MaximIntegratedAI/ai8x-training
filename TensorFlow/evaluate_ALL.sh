#!/bin/sh
echo "\n\nmnist:\n"
./evaluate_mnist.sh
echo "\n\ncifar10:\n"
./evaluate_cifar10.sh
echo "\n\ncifar100:\n"
./evaluate_cifar100.sh
echo "\n\nfashionmnist:\n"
./evaluate_fashionmnist.sh
echo "\n\nkws20:\n"
./evaluate_kws20.sh
echo "\n\nrock:\n"
./evaluate_rock.sh