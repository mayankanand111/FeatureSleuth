from dataclasses import dataclass

@dataclass
class params:
    train_data_path: str
    train_labels_path: str
    test_data_path: str
    test_labels_path: str

@dataclass
class hyperparams:
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str

@dataclass
class MNISTConfig:
    params: params