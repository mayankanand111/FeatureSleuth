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

@dataclass
class MNISTConfig:
    params: params