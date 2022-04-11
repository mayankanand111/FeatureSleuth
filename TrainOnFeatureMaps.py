import hydra
from hydra.core.config_store import ConfigStore

from Conf.DataConfig import MNISTConfig
from DataLoader import Loader
from DataLoader.Loader import FashionMNISTLoader
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel, BaseModelFeatureMap, FashionMnistModel
from Optimization.TrainingLoop import TrainLoop

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)


@hydra.main(config_path="Conf", config_name="DataConfig")
def test_mnist_dataset(cfg: MNISTConfig) -> None:
    # loading train loader to extract feature maps
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                          cfg.hyperparams.batch_size)

    # loading test dataset
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)

    # load the existing model
    mnist_base_model = BaseModel()

    TrainLoop.Tloop(mnist_base_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    train_loader, test_loader)

    mnist_feature_model = BaseModelFeatureMap()
    mnist_feature_model.load_state_dict(mnist_base_model.state_dict())
    feature_extraction_layers = ['conv1']
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, mnist_base_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=True)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    # calling Training Loop
    TrainLoop.Tloop(mnist_feature_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    feature_loader, test_loader, mnist_base_model, get_final_accuracy=False)

    del mnist_base_model
    model_to_clone_test_feature = BaseModel()
    model_to_clone_test_feature.load_state_dict(mnist_feature_model.state_dict())
    print("==================================")
    print("Final Accuracy Calculated on train set with feature map trained:")
    print(Evaluation.Eval(model_to_clone_test_feature, cfg.hyperparams.epochs - 1, train_loader, True))
    print("Final Accuracy Calculated on test set with feature map trained:")
    print(Evaluation.Eval(model_to_clone_test_feature, cfg.hyperparams.epochs - 1, test_loader, True))
    return


@hydra.main(config_path="Conf", config_name="DataConfig")
def test_fashion_mnist_dataset(cfg: MNISTConfig) -> None:
    # loading train loader to extract feature maps
    fashion_mnist_loader = FashionMNISTLoader()
    train_loader, test_loader = fashion_mnist_loader.load_test_and_trainset(cfg.hyperparams.batch_size)
    # load the existing model
    fashion_mnist_model = FashionMnistModel(padding=1)
    TrainLoop.Tloop(fashion_mnist_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    train_loader, test_loader)

    fashion_mnist_feature_model = FashionMnistModel(padding=2)
    fashion_mnist_feature_model.load_state_dict(fashion_mnist_model.state_dict())
    feature_extraction_layers = ['layer1.0']
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, fashion_mnist_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    # calling Training Loop
    TrainLoop.Tloop(fashion_mnist_feature_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate, feature_loader, test_loader, fashion_mnist_model,
                    get_final_accuracy=False)

    del fashion_mnist_model

    model_for_testing_fashion_mnist = FashionMnistModel(padding=1)
    model_for_testing_fashion_mnist.load_state_dict(fashion_mnist_feature_model.state_dict())
    print("==================================")
    print("Final Accuracy Calculated on train set with feature map trained:")
    print(Evaluation.Eval(model_for_testing_fashion_mnist, cfg.hyperparams.epochs - 1, train_loader, True))
    print("Final Accuracy Calculated on test set with feature map trained:")
    print(Evaluation.Eval(model_for_testing_fashion_mnist, cfg.hyperparams.epochs - 1, test_loader, True))

    return


if __name__ == "__main__":
    # test_mnist_dataset()
    test_fashion_mnist_dataset()
