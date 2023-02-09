import torch
from torch import nn

class MNIST_Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
    self.relu1 = nn.ReLU()
    
    self.conv2 = nn.Conv2d(32 , 16 , kernel_size = 3)
    self.relu2 = nn.ReLU()

    self.conv3 = nn.Conv2d(16 , 8 , kernel_size = 3)
    self.relu3 = nn.ReLU()
    
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.flatten= nn.Flatten() 
    self.fc1 = nn.Linear(in_features = 3872 , out_features= 10)

  def forward(self, input):
    y = self.conv1(input)
    y = self.relu1(y)

    y = self.conv2(y)
    y = self.relu2(y)
    
    y = self.conv3(y)
    y = self.relu3(y)
    
    y = self.pool1(y)
    y = self.flatten(y)
    y = self.fc1(y)
    return y