import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel ,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15 ,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30 ,kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480 ,64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64 ,10)
        self.softmax = nn.LogSoftmax()

    def forward(self ,finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput ,start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput


class BaseModelFeatureMap(nn.Module):
    def __init__(self):
        super(BaseModelFeatureMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput


class ThreeLayerModel(nn.Module):
    def __init__(self):
        super(ThreeLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = F.relu(self.pool3(self.conv3(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput

class ThreeLayerModelFeatureMap(nn.Module):
    def __init__(self):
        super(ThreeLayerModelFeatureMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = F.relu(self.pool3(self.conv3(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput