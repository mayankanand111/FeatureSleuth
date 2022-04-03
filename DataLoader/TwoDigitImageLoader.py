import gzip

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from DataLoader.Loader import MNISTDataset

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     # transforms.Normalize((0.5, ), (0.5, ))
     ])


class Two_Digit_Image_Train_Loader:
    def __init__(self):
        self

    def load_train_dataset(self, datapath, labelpath, batch_size, digit_one, digit_two, shuffle=True):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        train_data_images = np.reshape(train_data_images, (60000, 28, 28))
        print((train_data_images[0].shape))

        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        # train_label = np.frombuffer(buf, dtype=np.uint8, offset=8)
        train_label = np.frombuffer(buf, dtype=np.uint8, offset=8)
        # print(train_label)
        digit_one_indices = np.where(train_label == digit_one)
        digit_two_indices = np.where(train_label == digit_two)
        train_data_images_new = np.concatenate((np.take(train_data_images, digit_one_indices, 0)[0],
                                                np.take(train_data_images, digit_two_indices, 0)[0]))
        # print(train_data_images_new.shape)
        train_labels_new = np.concatenate((np.take(train_label, digit_one_indices, 0)[0],
                                           np.take(train_label, digit_two_indices, 0)[0]))
        train_labels_new[train_labels_new == digit_one] = 0
        train_labels_new[train_labels_new == digit_two] = 1


        train_data = MNISTDataset(train_data_images_new, train_labels_new, transform)
        #
        train_loader = DataLoader(train_data, batch_size, shuffle)
        return train_loader

    def load_test_dataset(self, datapath, labelpath, batch_size, digit_one, digit_two, shuffle=True):
        file_reader = gzip.open(datapath, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 10000)
        test_data_images = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        test_data_images = np.reshape(test_data_images, (10000, 28, 28))

        file_reader = gzip.open(labelpath, 'r')
        buf = file_reader.read()
        test_label = np.frombuffer(buf, dtype=np.uint8, offset=8)

        digit_one_indices = np.where(test_label == digit_one)
        digit_two_indices = np.where(test_label == digit_two)
        # print(len(np.take(test_data_images, digit_one_indices, 0)[0]))
        # print(len(np.take(test_data_images, digit_two_indices, 0)[0]))
        test_data_images_new = np.concatenate((np.take(test_data_images, digit_one_indices, 0)[0],
                                               np.take(test_data_images, digit_two_indices, 0)[0]))
        # print(test_data_images_new.shape)
        test_labels_new = np.concatenate((np.take(test_label, digit_one_indices, 0)[0],
                                          np.take(test_label, digit_two_indices, 0)[0]))

        test_labels_new[test_labels_new == digit_one] = 0
        test_labels_new[test_labels_new == digit_two] = 1

        # print(test_labels_new)

        test_data = MNISTDataset(test_data_images_new, test_labels_new, transform)

        test_loader = DataLoader(test_data, batch_size, shuffle)
        return test_loader
