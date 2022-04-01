import os

import torch
import torch.nn as nn
from PIL import Image as im
from numpy import savez_compressed
import numpy as np

from DataLoader import Loader
from FeatureExtracter.FeatureMapsExtract import FmapExtract
from FeatureExtracter.FeatureMapsExtractCosine import FeatureMapExtractor
from Models.model import BaseModel
from Visualization.LossCurve import LossCurve


def train_model(model, images, labels, loss_criterion, optimizer):
    model.train()
    # forward pass
    outputs = model(images)
    # finding loss
    loss = loss_criterion(outputs, labels)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_feature_maps(path, label):
    data = np.load("/Users/rohinichandrala/Documents/Personal_Docs/Dalhousie/ML/Project/featuresleuth/output/0/123.npz")
    for i in range(15):
        print(type(data['key_' + str(i)]))


def prepare_feature_map_dictionary(label, feature_maps, f_dict):
    # <class 'torch.Tensor'>
    # torch.Size([64, 15, 24, 24])

    for map, l in zip(feature_maps, label):
        key = str(l.item())
        value = f_dict.get(key)
        if value:
            print((map.detach().cpu().numpy()).shape)
        else:
            print("asdfasdfasdf")

    return f_dict
        #
        # path = "/Users/rohinichandrala/Documents/Personal_Docs/Dalhousie/ML/Project/featuresleuth/output/" + str(
        #     l.item())
        # os.makedirs(os.path.dirname(path), exist_ok=True)

    # f_dict = {}
    # for i in range(len(feature_maps)):
    #     f_dict['key_' + str(i)] = feature_maps[i].detach().numpy()

    # np.savez_compressed(path + '/123', **f_dict)

    #
    #     image = im.fromarray(feature_maps[i].detach().numpy())
    #     image.save('conv1_'+str(label.item())+'.png', format='PNG')
    #
    # with open(path, "a") as f:
    #     f.write(str(feature_maps))


class CosineTrain:
    def __init__(self):
        self

    def train_with_non_similar_images(self, model, epochs, optimizer, learning_rate, train_data_path, train_labels_path,
                                      test_data_path, test_labels_path, batch_size):
        train_loader = Loader.Train_Loader.load_train_dataset(train_data_path, train_labels_path, batch_size,
                                                              shuffle=False)
        stop_training_at_index = len(train_loader.dataset) / batch_size
        test_loader = Loader.Test_Loader.load_test_dataset(test_data_path, test_labels_path, batch_size, shuffle=False)

        # loss criteria
        loss_criterion = nn.NLLLoss()

        if optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), learning_rate)

        counter = 0
        f_dict = {}
        for images, label in train_loader:
            fmap_extractor = FeatureMapExtractor()
            feature_map_dict = fmap_extractor.extract(images, model)
            f_dict = prepare_feature_map_dictionary(label, feature_map_dict[0], f_dict)
            counter += 1

            if counter == 2:
                break

        print("Ended")
        loss_values = []
        # todo: comment this line after experiment
        epochs = 1
        # for epoch in range(epochs):
        #     useful_image_indices = []
        #     running_loss = 0
        #     counter = 0
        #     for images, labels in train_loader:
        #         # if counter <= stop_training_at_index:
        #         if counter <= 1:
        #             loss = train_model(model, images, labels, loss_criterion, optimizer)
        #             # save the index
        #
        #             # store the feature maps
        #
        #             feature_maps = FmapExtract.extract_featuremaps(images, model)
        #             print(type(feature_maps))
        #
        #         else:
        #             break
        #             # check if that is similar image
        #             # are_images_similar(images)
        #             # if False:
        #             #     loss = train_model(model, images, labels, loss_criterion, optimizer)
        #             # save the index
        #             # store the feature maps
        #
        #         counter += 1
        #
        #         running_loss += loss.item()
        #     loss_values.append(running_loss / len(train_loader))

        # evaluation of model on test data
        # if (cloned_model == None):
        #     Evaluation.Eval(model, epoch, test_loader)
        # else:
        #     cloned_model.load_state_dict(
        #         model.state_dict())  # this is recquired so that new weights are tranfered for testing
        #     Evaluation.Eval(cloned_model, epoch, test_loader)

        # Plotting Loss Curve
        # LossCurve.PlotCurve(loss_values, epochs)
