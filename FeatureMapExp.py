import torch
from DataLoader import Loader, TwoDigitImageLoader
import hydra
from hydra.core.config_store import ConfigStore

from FeatureExtracter.FeatureMapsExtractCosine import FeatureMapExtractor
from Models.model import BaseModel, ThreeLayerModel, BaseModelWithTwoDigits
from Optimization.CosineTrain import CosineTrain
from Optimization.TrainingLoop import TrainLoop
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
from Research.DataAnalysis import LoadData

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


def append_to_dataset(f_map, feature_map_dataset_dict, labels):
    if len(feature_map_dataset_dict.keys()) == 0:
        for key in f_map.keys():
            tensor_ds_new = Loader.TensorDataset
            tensor_ds_new.append(f_map.get(key), labels)
            feature_map_dataset_dict[key] = tensor_ds_new
    else:
        for key in f_map.keys():
            tensor_ds = feature_map_dataset_dict.get(key)
            tensor_ds.append(f_map.get(key))
            torch.cat(feature_map_dataset_dict.get(key), f_map.get(key), labels)
            feature_map_dataset_dict[key] = tensor_ds

    return feature_map_dataset_dict


@hydra.main(config_path="Conf", config_name="DataConfig")
def experiment(cfg: MNISTConfig) -> None:
    # take the images that are 7 or 8
    loader_instance = TwoDigitImageLoader.Two_Digit_Image_Train_Loader()
    train_loader = loader_instance.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                      cfg.hyperparams.batch_size, 7, 8, True)
    test_loader = loader_instance.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                    cfg.hyperparams.batch_size, 7, 8, True)

    # train the model
    model1 = BaseModelWithTwoDigits()
    TrainLoop.Tloop(model1, cfg.hyperparams.epochs, cfg.hyperparams.optimizer, cfg.hyperparams.learning_rate,
                    train_loader, test_loader)
    # get the accuracy here
    # extract feature maps of images using the trained model
    feature_map_dataset_dict = {}
    f_extractor = FeatureMapExtractor()
    for images, labels in train_loader:
        f_map = f_extractor.extract(images, model1)
        feature_map_dataset_dict = append_to_dataset(f_map, feature_map_dataset_dict, labels)

    model2 = BaseModelWithTwoDigits()
    for key in feature_map_dataset_dict.keys():
        f_train_loader = feature_map_dataset_dict.get(key)
        TrainLoop.Tloop(model2, cfg.hyperparams.epochs, cfg.hyperparams.optimizer, cfg.hyperparams.learning_rate,
                        f_train_loader, test_loader)

    # get the accuracy here


if __name__ == "__main__":
    experiment()
