
import gzip
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     #transforms.Normalize((0.5, ), (0.5, ))
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
        data = np.asarray(data).reshape(28, 28)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


class Train_Loader:
    def __init__(self):
        self
    def load_train_dataset(datapath,labelpath,batch_size):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        train_data_images = np.reshape(train_data_images, (60000, 784))

        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        train_label = np.frombuffer(buf, dtype=np.uint8, offset=8)

        train_data = MNISTDataset(train_data_images, train_label, transform)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        return train_loader