from dataclasses import dataclass

@dataclass
class params:
    train_data_path: str
    train_labels_path: str

@dataclass
class MNISTConfig:
    params: params