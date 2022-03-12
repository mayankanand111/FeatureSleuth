import gzip
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt


class LoadData:

    def __init__(self):
        self

    def Load_TrainData():
        file_reader = gzip.open('../featuresleuth/MNISTDataset/MNIST_train_data.gz', 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        train_data = np.reshape(train_data,(60000,784))
        return train_data

    def Load_TrainDataLabels():
        file_reader = gzip.open('../featuresleuth/MNISTDataset/MNIST_train_labels.gz', 'r')
        buf = file_reader.read()
        train_data = np.frombuffer(buf, dtype=np.uint8,offset=8)
        return train_data

    def PrintImage(data):
        test = np.reshape(data,(28,28))
        image = im.fromarray(np.reshape(data,(28,28)))
        image.show()

    def get_seperate_digits(train_data,train_data_labels):
        number_list = []
        dataset = np.hstack((train_data,train_data_labels.reshape(60000,1)))
        print(dataset[-1].shape[0]-1)
        dataset = dataset[dataset[:, dataset[-1].shape[0]-1].argsort()]
        dataset = np.split(dataset, np.where(np.diff(dataset[:, -1]))[0] + 1)
        return  dataset


    def find_digits_average(dataset_list):
        average_list = []
        for data in dataset_list:
            meanvalue_array = np.mean(data, keepdims=True, axis=0)
            meanvalue_array = meanvalue_array[:,:-1]
            average_list.append(meanvalue_array)
            f = plt.figure()
        for image,i in zip(average_list,range(10)):
            averaged_image = im.fromarray(np.reshape(image, (28, 28)))
            f.add_subplot(1, 10, i + 1)
            plt.imshow(averaged_image)
        plt.show(block=True)
        return average_list


    def find_euclediandistance(average_list,dataset_list):
        euledian_datalist = []
        for data,centroid in zip(dataset_list,average_list):
            data = data[:,:-1]
            euclidean_distance = np.sqrt(np.mean((data - centroid) ** 2,axis=1))
            data = np.column_stack((data,np.array(euclidean_distance)))
            data = data[data[:, data[-1].shape[0]-1].argsort()]
            euledian_datalist.append(data)
        return  euledian_datalist

    def get_mostdifferent_images(euledian_datalist):
        for data in euledian_datalist:
            top_5 = data[::-1][:5]
            top_5 = top_5[:,:-1]
            f = plt.figure()
            for image,i in zip(top_5,range(5)):
                different_image = im.fromarray(np.reshape(image, (28, 28)))
                f.add_subplot(1, 5, i + 1)
                plt.imshow(different_image)
            plt.show(block=True)

    def print_averagedImage(data):
        meanvalue  = np.mean(data,keepdims=True,axis=1)
        data = np.where((data>0),meanvalue,data)
        image = im.fromarray(np.reshape(data, (28, 28)))
        image.show()

    def __call__(self, ):
        return self.Load_Data()
