import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from .utils import DatasetNum

def cifar10_loaders( batch_size, data_root, num_workers=2, data_num=0,
        test_batch_size=100,
        transform_train=None, transform_test=None,
         ):

    if( transform_train is None ):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if( transform_test is None ):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    if( data_num > 0 ):
        trainset = DatasetNum( trainset, data_num )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return (trainloader, testloader)
