import hydra
from hydra.core.config_store import ConfigStore

from Conf.DataConfig import MNISTConfig
from DataLoader import Loader
from DataLoader.Loader import FashionMNISTLoader
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel, BaseModelFeatureMap, FashionMnistBaseModel
from Optimization.TrainingLoop import TrainLoop

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)

'''
This file has the experiments related to training model on feature maps.
There are two datasets - MNIST and fashion MNIST. For these two datasets two methods
are written. Before running this file, uncomment the method(which specifies the dataset)
that you want that is the end of the file.  
To run the file rightclick->Run 'TrainOnFeatureMaps'. 
As a part of this experiment, we will train a base model on a dataset(either MNIST 
or Fashion MNIST), extract feature maps for all the images or part of the images based on 
a param value 'sum_up_feature_channels', and transfer the weights to feature map model and 
train this feature map model on the feature maps generated. We then evaluate both the models
against the test set.
Note: Fashion MNIST dataset will be downloaded from the internet.
'''


@hydra.main(config_path="Conf", config_name="DataConfig")
def test_mnist_dataset(cfg: MNISTConfig) -> None:
    print("Experiment on MNIST Dataset")
    print(f'Batch-size: {cfg.hyperparams.batch_size}')
    print(f'Learning-rate: {cfg.hyperparams.learning_rate}')
    # loading train loader to extract feature maps
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                          cfg.hyperparams.batch_size)

    # loading test dataset
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)

    # load the existing model
    mnist_base_model = BaseModel()
    # path = cfg.params.pretrain_model_path
    # mnist_base_model.load_state_dict(torch.load(path + mnist_base_model.__class__.__name__))

    TrainLoop.Tloop(mnist_base_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    train_loader, test_loader)
    #
    # # Saving model trained weights
    # path = cfg.params.pretrain_model_path
    # torch.save(mnist_base_model.state_dict(), path + mnist_base_model.__class__.__name__)
    # print('Trained model : {} saved at {path}'.format(mnist_base_model.__class__.__name__, path=path))

    feature_extraction_layers = ['conv1']
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, mnist_base_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)
    mnist_feature_model = BaseModelFeatureMap()
    mnist_feature_model.load_state_dict(mnist_base_model.state_dict())
    # calling Training Loop
    TrainLoop.Tloop(mnist_feature_model, 2, cfg.hyperparams.optimizer,
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
    print("Experiment on Fashion MNIST Dataset")
    print(f'Batch-size: {cfg.hyperparams.batch_size}')
    print(f'Learning-rate: {cfg.hyperparams.learning_rate}')
    # loading train loader to extract feature maps
    fashion_mnist_loader = FashionMNISTLoader()
    train_loader, test_loader = fashion_mnist_loader.load_test_and_trainset(cfg.hyperparams.batch_size)
    # load the existing model
    fashion_mnist_model = FashionMnistBaseModel(padding=1)
    TrainLoop.Tloop(fashion_mnist_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    train_loader, test_loader)

    fashion_mnist_feature_model = FashionMnistBaseModel(padding=2)
    fashion_mnist_feature_model.load_state_dict(fashion_mnist_model.state_dict())
    feature_extraction_layers = ['layer1.0']
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, fashion_mnist_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    # calling Training Loop
    TrainLoop.Tloop(fashion_mnist_feature_model, 2, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate, feature_loader, test_loader, fashion_mnist_model,
                    get_final_accuracy=False)

    del fashion_mnist_model

    model_for_testing_fashion_mnist = FashionMnistBaseModel(padding=1)
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
