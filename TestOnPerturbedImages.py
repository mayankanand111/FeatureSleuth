import hydra
import torch
from hydra.core.config_store import ConfigStore

from Conf.DataConfig import MNISTConfig
from DataLoader import Loader
from DataLoader.Loader import PerturbImageLoader
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel, BaseModelFeatureMap
from Optimization.PerturbImageGenerator import PerturbImageGenerator
from Optimization.TrainingLoop import TrainLoop

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


@hydra.main(config_path="Conf", config_name="DataConfig")
def test_on_pertub_Images(cfg: MNISTConfig) -> None:
    # loading train loader to extract feature maps
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                          cfg.hyperparams.batch_size)

    # loading test dataset
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)

    # load the existing model
    mnist_base_model = BaseModel()

    print("------Training base model------")
    TrainLoop.Tloop(mnist_base_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    train_loader, test_loader)

    feature_extraction_layers = ['conv1']

    print("------Generating feature maps------")
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, mnist_base_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    mnist_feature_model = BaseModelFeatureMap()
    mnist_feature_model.load_state_dict(mnist_base_model.state_dict())

    print("------Generating pertubed images------")
    perturb_generator = PerturbImageGenerator()
    perturb_dataset = perturb_generator.get(mnist_base_model, train_loader)
    perturb_loader = PerturbImageLoader(perturb_dataset)
    perturb_data_loader = perturb_loader.get(cfg.hyperparams.batch_size)

    print("==================================")
    print("Testing model on perturbed images before training on feature maps ")
    print("Accuracy Calculated on perturbed Images(with epsilon = 0.05) ")
    print(Evaluation.Eval(mnist_base_model, cfg.hyperparams.epochs - 1, perturb_data_loader, False))

    print("------Training the base model on feature maps------")
    # calling Training Loop
    TrainLoop.Tloop(mnist_feature_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    feature_loader, test_loader, mnist_base_model, get_final_accuracy=False)


    del mnist_base_model
    model_to_clone_test_perturb = BaseModel()
    model_to_clone_test_perturb.load_state_dict(mnist_feature_model.state_dict())
    print("==================================")
    print("Testing new model on perturbed images ")
    print("Accuracy Calculated on perturbed Images(with epsilon = 0.05) ")
    print(Evaluation.Eval(model_to_clone_test_perturb, cfg.hyperparams.epochs - 1, perturb_data_loader, False))
    return


if __name__ == "__main__":
    test_on_pertub_Images()
