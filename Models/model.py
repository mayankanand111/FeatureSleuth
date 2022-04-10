import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480 ,64)
        self.fc2 = nn.Linear(64 ,10)
        self.softmax = nn.LogSoftmax()

    def forward(self ,finput):
        finput = torch.relu(self.pool1(self.conv1(finput)))
        finput = torch.relu(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput ,start_dim=1)
        finput = torch.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput

class BaseModelWithTwoDigits1(nn.Module):
    def __init__(self, padding_digits=0):
        super(BaseModelWithTwoDigits1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5, padding=padding_digits)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 3)
        # self.fc2 = nn.Linear(64, 1)
        self.softmax = nn.LogSoftmax()
        # self.softmax = nn.Sigmoid()

    def forward(self, finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput


class BaseModelWithSigmoid(nn.Module):
    def __init__(self):
        super(BaseModelWithSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480 ,64)
        self.fc2 = nn.Linear(64 ,10)
        self.softmax = nn.LogSoftmax()

    def forward(self ,finput):
        finput = torch.sigmoid(self.pool1(self.conv1(finput)))
        finput = torch.sigmoid(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput ,start_dim=1)
        finput = torch.sigmoid(self.fc1(finput))
        finput = self.fc2(finput)
        finput = self.softmax(finput)
        return finput

class BaseModelWithTwoDigits(nn.Module):
    def __init__(self, padding_digits=0):
        super(BaseModelWithTwoDigits, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5, padding=padding_digits)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(480, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, 1)
        # self.softmax = nn.Softmax()
        self.softmax = nn.Sigmoid()

    def forward(self, finput):
        finput = F.relu(self.pool1(self.conv1(finput)))
        finput = F.relu(self.pool2(self.conv2(finput)))
        finput = torch.flatten(finput, start_dim=1)
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(2420, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, finput):
        finput = F.relu((self.conv1(finput)))
        finput = F.relu((self.conv2(finput)))
        finput = F.relu(self.pool1(self.conv3(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = F.relu(self.fc2(finput))
        finput = self.fc3(finput)
        finput = self.softmax(finput)
        return finput


class ThreeLayerModelFeatureMap(nn.Module):
    def __init__(self):
        super(ThreeLayerModelFeatureMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(2420, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, finput):
        finput = F.relu((self.conv1(finput)))
        finput = F.relu((self.conv2(finput)))
        finput = F.relu(self.pool1(self.conv3(finput)))
        finput = torch.flatten(finput, start_dim=1)
        finput = F.relu(self.fc1(finput))
        finput = F.relu(self.fc2(finput))
        finput = self.fc3(finput)
        finput = self.softmax(finput)
        return finput

class CatAndDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        #         print(X.shape)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        #         X = torch.sigmoid(X)
        return X

class CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x