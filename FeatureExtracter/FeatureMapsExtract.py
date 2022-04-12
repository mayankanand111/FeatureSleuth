import numpy
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor


class FmapExtract:

    def extract_featuremaps(batch_images, model, feature_extraction_layers):
        # model_weights = []
        # conv_layers = []
        # model_children = list(model.children())
        # counter = 0
        # for i in range(len(model_children)):
        #     if type(model_children[i]) == nn.Conv2d:
        #         counter += 1
        #         model_weights.append(model_children[i].weight)
        #         conv_layers.append(model_children[i])
        #     elif type(model_children[i]) == nn.Sequential:
        #         for j in range(len(model_children[i])):
        #             for child in model_children[i][j].children():
        #                 if type(child) == nn.Conv2d:
        #                     counter += 1
        #                     model_weights.append(child.weight)
        #                     conv_layers.append(child)

        feature_extractor = create_feature_extractor(model, return_nodes=feature_extraction_layers)
        feature_map_images = feature_extractor(batch_images)

        return feature_map_images

    def getfeatures_from_loader(input_loader, model, feature_extraction_layers, sum_up_feature_channels=True):
        feature_maps = None
        train_labels = None
        # counter = 0
        # stop_counter_at = len(input_loader)/20
        for images, labels in input_loader:
            # if not sum_up_feature_channels and counter >= stop_counter_at:
            #     print("counter stopped at", counter)
            #     break
            feature_map_dict = FmapExtract.extract_featuremaps(images, model, feature_extraction_layers)
            for layer in feature_map_dict.keys():
                if sum_up_feature_channels:
                    if feature_maps is None:
                        size = feature_map_dict[layer][0].shape[1]
                        temp = torch.sum(feature_map_dict[layer].flatten(start_dim=2, end_dim=3), dim=1)
                        temp = temp.reshape(len(images), 1, size, size)
                        feature_maps = temp
                        train_labels = labels
                    else:
                        size = feature_map_dict[layer][0].shape[1]
                        temp = torch.sum(feature_map_dict[layer].flatten(start_dim=2, end_dim=3), dim=1)
                        temp = temp.reshape(len(images), 1, size, size)
                        feature_maps = torch.vstack((feature_maps, temp))
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
            # counter += 1
        return [t.detach().numpy() for t in feature_maps], np.array(train_labels, dtype=np.uint8)

    # def get_feature_maps(model, batch_images):
    #     feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
    #     newBatchFeatureMaps = feature_extractor(batch_images)
    #     print(newBatchFeatureMaps.shape)
    #
    #     Z = torch.sum(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), dim=1)
    #     # print(type(Z.shape))
    #     return Z