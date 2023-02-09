import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from time import time

def train_dataset(transform):
    train_data = MNIST('./data', download = True, train = True, transform=transform)
    return train_data

def test_dataset(transform):
    test_data  = MNIST('./data', download = False, train = False, transform=transform)
    return test_data