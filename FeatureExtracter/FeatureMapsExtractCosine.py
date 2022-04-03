import torch.nn as nn
import numpy as np


def get_feature_maps(conv_layer, batch_images):
    return conv_layer(batch_images)


class FeatureMapExtractor:
    def extract(self, batch_images, model):
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
        names = []
        featuremap_dict = {}

        # for index in range(len(conv_layers)):
        for index in range(1):
            featuremap_dict[str(index)] = get_feature_maps(conv_layers[index], batch_images)

        # # print("output length", len(outputs))
        # processed = []
        # for feature_map in outputs:
        #     print((feature_map))
        #     for f in range(feature_map.size()[1]):
        #         processed.append(feature_map[0][f])
        #
        # feature_map_image = []
        # for i in range(len(processed)):
        #     feature_map_image.append(processed[i])
        # # print(len(feature_map_image))
        # return feature_map_image, "conv1"
        return featuremap_dict
