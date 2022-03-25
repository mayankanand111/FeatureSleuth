import Models.model
import torch
from DataLoader import Loader
import hydra
from hydra.core.config_store  import ConfigStore

from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel,BaseModelFeatureMap,ThreeLayerModel,ThreeLayerModelFeatureMap
from Optimization.TrainingLoop import TrainLoop
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
from Research.DataAnalysis import LoadData


cs = ConfigStore.instance()
cs.store(name="mnsit_config",node=MNISTConfig)

@hydra.main(config_path="Conf", config_name="DataConfig")
def main(cfg: MNISTConfig) -> None:
    # Load Train and Test Loader
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path,cfg.params.train_labels_path,cfg.hyperparams.batch_size)
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path,cfg.params.test_labels_path,cfg.hyperparams.batch_size)

    # creating model
    loaded_model = BaseModel() # in this define model that you want to create
    model_to_clone = BaseModelFeatureMap()  # in this use model that you want to clone
    #assigning weights from pre trained model
    path = cfg.params.pretrain_model_path
    loaded_model.load_state_dict(torch.load(path+model_to_clone.__class__.__name__))

    # state_dict = torch.load(path+model_to_clone.__class__.__name__)
    # with torch.no_grad():
        # loaded_model.conv1.weight.copy_(state_dict['conv1.weight'])
        # loaded_model.conv1.bias.copy_(state_dict['conv1.bias'])
        # loaded_model.conv2.weight.copy_(state_dict['conv2.weight'])
        # loaded_model.conv2.bias.copy_(state_dict['conv2.bias'])
        # loaded_model.fc1.weight.copy_(state_dict['fc1.weight'])
        # loaded_model.fc1.bias.copy_(state_dict['fc1.bias'])
        # loaded_model.fc2.weight.copy_(state_dict['fc2.weight'])
        # loaded_model.fc2.bias.copy_(state_dict['fc2.bias'])
    print("-----------------Testing on {} model------------------".format(model_to_clone.__class__.__name__))
    # calling Eval function to test trained model
    Evaluation.Eval(loaded_model, 1, train_loader,True) # keep epoch 1 as we want to test only once on whole dataset
    Evaluation.Eval(loaded_model, 1, test_loader, True) # True as last argument shows you are  testing pretrained model
if __name__ == "__main__":
    main()