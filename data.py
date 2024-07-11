import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import random

def get_CIFAR10(validation_size = None):
    transform_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            transforms.Resize(40, antialias=True),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(p=0.5)
                        ])
    
    transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    validation_set = None
    if validation_size != None:
        np.random.seed(128)
        I = np.random.permutation(len(train_set))
        validation_set = Subset(train_set, I[:validation_size])
        train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set


