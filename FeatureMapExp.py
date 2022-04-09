import numpy
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor


class FmapExtract:

    def extract_featuremaps(batch_images, model, feature_extraction_layers):
        model_weights = []
        conv_layers = []
        model_children = list(model.children())
        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)

        feature_extractor = create_feature_extractor(model, return_nodes=feature_extraction_layers)
        feature_map_images = feature_extractor(batch_images)

        return feature_map_images

    def getfeatures_from_loader(input_loader, model, feature_extraction_layers, sum_up_feature_channels=True):
        feature_maps = None
        train_labels = None
        for images, labels in input_loader:
            feature_map_dict = FmapExtract.extract_featuremaps(images, model, feature_extraction_layers)
            for layer in feature_map_dict.keys():
                if sum_up_feature_channels:
                    if feature_maps is None:
                        feature_maps = torch.sum(feature_map_dict[layer], dim=1, keepdim=True)
                        train_labels = labels
                    else:
                        feature_maps = torch.vstack(
                            (feature_maps, torch.sum(feature_map_dict[layer], dim=1, keepdim=True)))
                        train_labels = torch.cat((train_labels, labels), 0)
                else:
                    temp = feature_map_dict[layer]
                    current_f = torch.reshape(temp, (len(temp) * temp.shape[1], 1, temp.shape[2], temp.shape[3]))
                    if feature_maps is None:
                        feature_maps = current_f
                    else:
                        feature_maps = torch.vstack((feature_maps, current_f))
                    if train_labels is None:
                        train_labels = labels.repeat_interleave(temp.shape[1])
                    else:
                        train_labels = torch.cat((train_labels, labels.repeat_interleave(temp.shape[1])))
        return [t.detach().numpy() for t in feature_maps], np.array(train_labels, dtype=np.uint8)

    def get_feature_maps(model, batch_images):
        feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
        newBatchFeatureMaps = feature_extractor(batch_images)
        print(newBatchFeatureMaps.shape)

        Z = torch.sum(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), dim=1)
        # print(type(Z.shape))
        return Z

    def extract_featuremaps_new(batch_images, model):
        model_weights = []
        conv_layers = []
        model_children = list(model.children())
        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)

        outputs = []
        for layer in conv_layers[0:1]:
            image = layer(batch_images)
            outputs.append(image)

        return outputs[0]
