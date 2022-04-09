import hydra
import torch
from hydra.core.config_store import ConfigStore

from Conf.DataConfig import MNISTConfig
from DataLoader import Loader
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel, BaseModelFeatureMap
from Optimization.TrainingLoop import TrainLoop

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


@hydra.main(config_path="Conf", config_name="DataConfig")
def main(cfg: MNISTConfig) -> None:
    # loading train loader to extract feature maps
    # keep batch size 1 here so that indivisual image comes.
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path, 256)

    # loading test loaderl
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)
    train_loader1 = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                           cfg.hyperparams.batch_size)

    # creating model
    loaded_model = BaseModelFeatureMap()
    model_to_clone = BaseModel()
    # assigning weights from pre trained model
    path = cfg.params.pretrain_model_path
    loaded_model.load_state_dict(torch.load(path + model_to_clone.__class__.__name__))
    model_to_clone.load_state_dict(torch.load(path + model_to_clone.__class__.__name__))
    feature_extraction_layers = ['conv1']
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, model_to_clone,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    # calling Training Loop
    TrainLoop.Tloop(loaded_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer, cfg.hyperparams.learning_rate,
                    feature_loader, test_loader, model_to_clone, get_final_accuracy=False)

    model_to_clone = BaseModel()
    model_to_clone.load_state_dict(torch.load(path + model_to_clone.__class__.__name__))
    print("==================================")
    print("Final Accuracy Calculated on train set:")
    Evaluation.Eval(model_to_clone, cfg.hyperparams.epochs - 1, train_loader1, True)
    print("Final Accuracy Calculated on test set:")
    Evaluation.Eval(model_to_clone, cfg.hyperparams.epochs - 1, test_loader)

    model_to_clone = BaseModel()
    model_to_clone.load_state_dict(loaded_model.state_dict())
    print("==================================")
    print("Final Accuracy Calculated on train set with featuremap trained:")
    Evaluation.Eval(model_to_clone, cfg.hyperparams.epochs - 1, train_loader1, True)
    print("Final Accuracy Calculated on test set with featuremap trained:")
    Evaluation.Eval(model_to_clone, cfg.hyperparams.epochs - 1, test_loader, True)

    # Saving model trained weights
    path = cfg.params.pretrain_model_path
    torch.save(loaded_model.state_dict(), path + loaded_model.__class__.__name__)
    print('Trained model : {} saved at {path}'.format(loaded_model.__class__.__name__, path=path))
    return


if __name__ == "__main__":
    main()
