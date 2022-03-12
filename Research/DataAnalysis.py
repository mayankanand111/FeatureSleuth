import gzip
import numpy as np
from PIL import Image as im


class LoadData:
    # change patch as per your file location
    file_path = '/Users/rohinichandrala/Documents/Personal_Docs/Dalhousie/ML/Project/featuresleuth/MNISTDataset/MNIST_train_data.gz'

    def __init__(self):
        self

    def Load_Data():
        file_reader = gzip.open(LoadData.file_path, 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        train_data = train_data.reshape(60000, 28 * 28)
        # print('demo')
        return train_data

    def PrintImage(data):
        image = im.fromarray(np.reshape(data, (28, 28)), mode=None)
        # image.show()
        return image

    def __call__(self, ):
        return self.Load_Data()


class LoadLabels:
    # change patch as per your file location
    label_file_path = '/Users/rohinichandrala/Documents/Personal_Docs/Dalhousie/ML/Project/featuresleuth/MNISTDataset/MNIST_train_labels.gz'

    def load_labels(self):
        train_labels = gzip.open(LoadLabels.label_file_path, 'rb')
        buf = train_labels.read()
        train_labels = np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int32)
        return train_labels
