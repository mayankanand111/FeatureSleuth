import hydra
import torch
from hydra.core.config_store import ConfigStore

from Conf.DataConfig import MNISTConfig
from DataLoader import Loader
from DataLoader.Loader import PerturbImageLoader, TensorDataset
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from ModelEvaluation.Evaluation import Evaluation
from Models.model import BaseModel, BaseModelFeatureMap
from Optimization.PerturbImageGenerator import PerturbImageGenerator
from Optimization.TrainingLoop import TrainLoop

cs = ConfigStore.instance()
cs.store(name="mnsit_config", node=MNISTConfig)

'''
This file has the experiments related to training model on feature maps and 
checking it on the perturbed images.
For this experiment, MNIST dataset is used. 
To run the file rightclick -> Run 'TestOnPerturbedImages'. 
As a part of this experiment, we will first train a base model on MNIST dataset. Then
feature maps are extracted for all the images or part of the images based on a param 
value 'sum_up_feature_channels', and then perturbed images are generated using the same 
model. Base model is first evaluated on the generated perturbed images and then
feature map model is trained on the feature maps. Once done, feature map model is evaluated 
on the generated perturbed images.
For generating perturbed images 10k of the train images are taken and the base model is trained
only on the remaining 50k images. Also feature map model is trained on the feature maps
extracted from these images.
'''

@hydra.main(config_path="Conf", config_name="DataConfig")
def test_on_pertub_Images(cfg: MNISTConfig) -> None:
    # loading train loader to extract feature maps
    train_loader, validation_loader = Loader.Train_Val_Loader.load_train_dataset(cfg.params.train_data_path,
                                                                                 cfg.params.train_labels_path,
                                                                                 cfg.hyperparams.batch_size)

    # loading test dataset
    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)

    epsilon = 0.25

    # load the existing model
    mnist_base_model = BaseModel()

    print("------Training base model------")
    TrainLoop.Tloop(mnist_base_model, cfg.hyperparams.epochs, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate, train_loader, test_loader)

    feature_extraction_layers = ['conv1']

    print("------Generating feature maps------")
    # extracting feature maps
    train_images, train_labels = FmapExtract.getfeatures_from_loader(train_loader, mnist_base_model,
                                                                     feature_extraction_layers,
                                                                     sum_up_feature_channels=False)
    feature_loader = Loader.Feature_loader.create_feature_loader(train_images, train_labels, cfg.hyperparams.batch_size)

    print("------Generating pertubed images------")
    perturb_generator = PerturbImageGenerator()
    # pertub_dataset_original = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
    #                                                    cfg.hyperparams.batch_size)

    # perturb_dataset = perturb_generator.get(mnist_base_model, train_loader)
    original_input, input_grad, original_labels = perturb_generator.get(mnist_base_model, validation_loader, epsilon=epsilon)
    product_value = torch.mul(epsilon, torch.sign(input_grad))
    perturbed_images = original_input.add(product_value)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    tensor_data = TensorDataset(perturbed_images, original_labels)
    perturb_loader = PerturbImageLoader(tensor_data)
    perturb_data_loader = perturb_loader.get(cfg.hyperparams.batch_size)

    print("==================================")
    print("Testing model on perturbed images before training on feature maps ")
    print(f'Accuracy Calculated on perturbed Images(with epsilon = {epsilon}) ')
    print(Evaluation.Eval(mnist_base_model, cfg.hyperparams.epochs - 1, perturb_data_loader, False))

    print("------Training the base model on feature maps------")
    mnist_feature_model = BaseModelFeatureMap()
    mnist_feature_model.load_state_dict(mnist_base_model.state_dict())
    # calling Training Loop
    TrainLoop.Tloop(mnist_feature_model, 2, cfg.hyperparams.optimizer,
                    cfg.hyperparams.learning_rate,
                    feature_loader, test_loader, mnist_base_model, get_final_accuracy=False)

    del mnist_base_model
    model_to_clone_test_perturb = BaseModel()
    model_to_clone_test_perturb.load_state_dict(mnist_feature_model.state_dict())
    print("==================================")
    print("Testing new model on perturbed images ")
    print(f'Accuracy Calculated on perturbed Images(with epsilon = {epsilon}) ')
    print(Evaluation.Eval(model_to_clone_test_perturb, cfg.hyperparams.epochs - 1, perturb_data_loader, False))
    return


if __name__ == "__main__":
    test_on_pertub_Images()
