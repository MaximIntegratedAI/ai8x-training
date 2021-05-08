###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Utility functions for NAS
"""

import torch


def calc_accuracy(child_net_arch, model, train_loader, test_loader, device):
    """Calculates accuracy for the given subnet of the model"""
    correct = 0
    total = 0

    with torch.no_grad():
        if child_net_arch is not None:
            model.set_subnet_arch(child_net_arch, True)

        if model.bn:
            model.train()
            if train_loader:
                for data in train_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
            else:
                for _ in range(5):
                    for data in test_loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)

        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_accuracy = correct / total

        if child_net_arch is not None:
            model.reset_arch(True)

    return val_accuracy


def calc_efficiency(child_net_arch):  # pylint: disable=unused-argument
    """Calculates efficiency for the given subnet of the model"""
    return 1.0


def check_net_in_population(child_net, population):
    """Checks if a sub network is in the population"""
    for p in population:
        if child_net == p[0]:
            return True

    return False
