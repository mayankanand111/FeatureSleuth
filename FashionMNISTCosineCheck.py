import hydra
import torchvision
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from Conf.DataConfig import MNISTConfig
from Models.model import BaseModel, FashionMnistModel
from Optimization.TLoopWithExtraction import TLoopWithExtraction
from Optimization.TrainingLoop import TrainLoop

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

    #creating model
    model = FashionMnistModel()

    #calling Training Loop
    TrainLoop.Tloop(model,cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,train_loader,test_loader)
    del model

    model = FashionMnistModel()
    firstlayername = "layer1"
    TLoopWithExtraction.Tloop_Extraction(model,32,firstlayername, cfg.hyperparams.epochs,cfg.hyperparams.optimizer,cfg.hyperparams.learning_rate,train_loader,test_loader)
    del model
    
    # # Saving model trained weights
    # path = cfg.params.pretrain_model_path
    # torch.save(model.state_dict(), path + model.__class__.__name__)
    # print('Trained model : {} saved at {path}'.format(model.__class__.__name__, path=path))


if __name__ == "__main__":
    main()
