import gzip
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data

'''
This file has all the data loaders used in various experiments.
'''
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     # transforms.Normalize((0.5, ), (0.5, ))
     ])


class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        data = np.asarray(data)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data.float(), self.y[i])
        else:
            return data.float()


class TensorDataset(Dataset):
    def __init__(self, tensors, labels=None):
        # tensors = torch.reshape(tensors, (len(tensors), 1, 24, 24))
        # print(tensors.shape)
        self.X = tensors
        self.y = labels

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        # data = np.asarray(data)

        # data = transform(data)
        # print("Inside tensor loader", data.shape)

        if self.y is not None:
            return (data.float(), self.y[i])
        else:
            return data.float()

    def append(self, tensors, labels):
        # tensors = torch.reshape(tensors, (len(tensors), 1, 24, 24))
        # print(tensors.shape)
        if self.X is None:
            self.X = tensors
            self.y = labels
        else:
            self.X = torch.cat((self.X, tensors), 0)
            self.y = torch.cat((self.y, labels), 0)


class FeatureDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        data = np.asarray(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


class Train_Val_Loader:

    def __init__(self):
        self

    def load_train_dataset(datapath, labelpath, batch_size, shuffle=True):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        train_data_images = np.reshape(train_data_images, (60000, 28, 28))
        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        train_label = np.frombuffer(buf, dtype=np.uint8, offset=8)

        train_data = MNISTDataset(train_data_images, train_label, transform)
        train_set, validation_set = data.random_split(train_data, [50000, 10000])

        train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
        validation_loader = DataLoader(validation_set, batch_size, shuffle=True, drop_last=True)
        return train_loader, validation_loader


class Train_Loader:
    def __init__(self):
        self

    def load_train_dataset(datapath, labelpath, batch_size, shuffle=True):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        train_data_images = np.reshape(train_data_images, (60000, 28, 28))

        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        train_label = np.frombuffer(buf, dtype=np.uint8, offset=8)

        train_data = MNISTDataset(train_data_images, train_label, transform)

        train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=False)
        return train_loader


class Test_Loader:
    def __init__(self):
        self

    def load_test_dataset(datapath, labelpath, batch_size, shuffle=True):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 10000)
        test_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        test_data_images = np.reshape(test_data_images, (10000, 28, 28))

        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        test_label = np.frombuffer(buf, dtype=np.uint8, offset=8)

        test_data = MNISTDataset(test_data_images, test_label, transform)

        test_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=False)
        return test_loader


class FashionMNISTLoader:
    def __init__(self):
        self

    def load_test_and_trainset(self, batch_size, shuffle=True):
        T = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        train_data = torchvision.datasets.FashionMNIST("./data/mnist/train_data", train=True, download=True,
                                                       transform=T)
        test_data = torchvision.datasets.FashionMNIST("./data/mnist/test_data", train=False, download=True, transform=T)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, test_loader


class PerturbImageLoader:
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def get(self, batch_size, shuffle=True):
        perturb_image_loader = DataLoader(self.tensor_dataset, batch_size, shuffle)
        return perturb_image_loader


class Feature_loader:
    def __init__(self):
        self

    def create_feature_loader(train_images, train_labels, batch_size, shuffle=True):
        feature_data = FeatureDataset(train_images, train_labels)
        feature_loader = DataLoader(feature_data, batch_size, shuffle)
        return feature_loader
