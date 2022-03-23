import torch
from DataLoader import Loader
import hydra
from hydra.core.config_store  import ConfigStore

from FeatureExtracter.FeatureMapsExtract import FmapExtract
from Models.model import BaseModel,BaseModelFeatureMap
from Optimization.TrainingLoop import TrainLoop
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
from Research.DataAnalysis import LoadData

cs = ConfigStore.instance()
cs.store(name="mnsit_config",node=MNISTConfig)

@hydra.main(config_path="Conf", config_name="DataConfig")
def main(cfg: MNISTConfig) -> None:

    #loading train loader to extract feature maps
    #keep batch size 1 here so that indivisual image comes.
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path,cfg.params.train_labels_path,1)

    #loading test loader
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path,cfg.params.test_labels_path,cfg.hyperparams.batch_size)

    # creating model
    loaded_model = BaseModelFeatureMap()
    model_to_clone = BaseModel()
    #assigning weights from pre trained model
    path = cfg.params.pretrain_model_path
    loaded_model.load_state_dict(torch.load(path+model_to_clone.__class__.__name__))

    #extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader,model_to_clone)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images,train_labels,cfg.hyperparams.batch_size)

    # list = []
    # for images,labels in test_loader:
    #     list.append(images)
    #
    # list = []
    # for images,labels in feature_loader:
    #     list.append(images)

    # calling Training Loop
    TrainLoop.Tloop(loaded_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer, cfg.hyperparams.learning_rate,
                    feature_loader, test_loader,model_to_clone)

    # Saving model trained weights
    path = cfg.params.pretrain_model_path
    torch.save(loaded_model.state_dict(), path + loaded_model.__class__.__name__)

if __name__ == "__main__":
    main()