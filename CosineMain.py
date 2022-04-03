import torch
from DataLoader import Loader
import hydra
from hydra.core.config_store import ConfigStore
from Models.model import BaseModel, ThreeLayerModel
from Optimization.CosineTrain import CosineTrain
from Optimization.TrainingLoop import TrainLoop
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
from Research.DataAnalysis import LoadData

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


@hydra.main(config_path="Conf", config_name="DataConfig")
def run_experiment(cfg: MNISTConfig) -> None:
    # Load Train and Test Loader for training the model
    # train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
    #                                                       cfg.hyperparams.batch_size)
    # test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
    #                                                    cfg.hyperparams.batch_size)
    #
    # # Training without eliminating the images
    # model = BaseModel()
    # TrainLoop.Tloop(model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer, cfg.hyperparams.learning_rate,
    #                 train_loader, test_loader)

    # creating model
    cosine_model = BaseModel()
    cosine_train_inst = CosineTrain()

    model_to_load = BaseModel()
    trained_model_copy = BaseModel()
    # assigning weights from pre trained model
    path = cfg.params.pretrain_model_path
    trained_model_copy.load_state_dict(torch.load(path + model_to_load.__class__.__name__))

    # calling Training Loop
    cosine_train_inst.train_with_non_similar_images(trained_model_copy, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                                                    cfg.hyperparams.learning_rate, cfg.params.train_data_path,
                                                    cfg.params.train_labels_path, cfg.params.test_data_path,
                                                    cfg.params.test_labels_path, cfg.hyperparams.batch_size)
    # cosine_train_inst.train_with_non_similar_images(trained_model_copy, cfg.hyperparams.epochs,
    #                                                 cfg.hyperparams.optimizer,
    #                                                 cfg.hyperparams.learning_rate, cfg.params.train_data_path,
    #                                                 cfg.params.train_labels_path, cfg.params.test_data_path,
    #                                                 cfg.params.test_labels_path, 1)


if __name__ == "__main__":
    run_experiment()
