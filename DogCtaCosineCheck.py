from lib2to3.pytree import Base

import torchvision.transforms as transforms
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

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.hyperparams.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.hyperparams.batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    # # creating model
    # model = BaseModel()
    #
    # #calling Training Loop
    # TrainLoop.Tloop(model,cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,train_loader,test_loader)

    model = Models.model.CIFAR()

    TLoopWithExtraction.Tloop_Extraction(model,cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,trainloader,testloader)

    # # Saving model trained weights
    # path = cfg.params.pretrain_model_path
    # torch.save(model.state_dict(), path + model.__class__.__name__)
    # print('Trained model : {} saved at {path}'.format(model.__class__.__name__, path=path))


if __name__ == "__main__":
    main()
