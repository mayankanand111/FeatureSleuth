
import gzip
import numpy as np

class Train_Loader:
    def __init__(self):
        self
    def load_train_dataset(self):
        file_reader = gzip.open('../featuresleuth/MNISTDataset/MNIST_train_data.gz', 'r')
        file_reader.read(16)
        buf = file_reader.read(28 * 28 * 60000)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)

        #train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)