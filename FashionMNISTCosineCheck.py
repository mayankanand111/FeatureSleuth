from lib2to3.pytree import Base
import Models.model
import torch
from DataLoader import Loader
import hydra
from hydra.core.config_store  import ConfigStore
from Models.model import BaseModel, BaseModelWithSigmoid,ThreeLayerModel
from Optimization.TLoopWithExtraction import TLoopWithExtraction
from Optimization.TrainingLoop import TrainLoop
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
from Research.DataAnalysis import LoadData
import torchvision
from torch.utils.data import DataLoader, random_split


cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


@hydra.main(config_path="Conf", config_name="DataConfig")
def main(cfg: MNISTConfig) -> None:
    # Load Train and Test Loader

    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
])

    train_data = torchvision.datasets.FashionMNIST("./data/mnist/train_data", train= True, download = True, transform = T)
    test_data = torchvision.datasets.FashionMNIST("./data/mnist/test_data", train= False, download = True, transform = T)

    train_loader = DataLoader(train_data, batch_size = cfg.hyperparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = cfg.hyperparams.batch_size, shuffle=True)

    # creating model
    model = BaseModel()
    
    #calling Training Loop
    TrainLoop.Tloop(model,cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,train_loader,test_loader)
    del model

    model = BaseModel()

    TLoopWithExtraction.Tloop_Extraction(model,cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,train_loader,test_loader)
    del model
    
    # # Saving model trained weights
    # path = cfg.params.pretrain_model_path
    # torch.save(model.state_dict(), path + model.__class__.__name__)
    # print('Trained model : {} saved at {path}'.format(model.__class__.__name__, path=path))


if __name__ == "__main__":
    main()
