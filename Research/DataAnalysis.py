import gzip
import numpy as np
from PIL import Image as im

class LoadData:

    def __init__(self):
        self

    def Load_Data():
        #change patch as per your file location
        file_reader = gzip.open('/Users/lib-user/Library/CloudStorage/OneDrive-DalhousieUniversity/Winter 2022/Machine Learning/featuresleuth/MNISTDataset/MNIST_train_data.gz', 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 1)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        print('demo')
        return train_data

    def PrintImage(data):
        image = im.fromarray(np.reshape(data,(28,28)))
        image.show()

    def __call__(self, ):
        return self.Load_Data()
