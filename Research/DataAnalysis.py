import gzip
import numpy as np


class LoadData:
    train_data = None

    def __init__(self):
        self

    def Load_Data():
        #change patch as per your file location
        train_data = gzip.open('/Users/lib-user/Library/CloudStorage/OneDrive-DalhousieUniversity/Winter 2022/Machine Learning/featuresleuth/MNISTDataset/MNIST_train_data.gz', 'r')
        print('demo')
        return train_data

    def __call__(self, ):
        return self.Load_Data()
