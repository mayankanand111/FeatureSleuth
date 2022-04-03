import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


def get_feature_maps(model, batch_images):
    feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
    newBatchFeatureMaps = feature_extractor(batch_images)
    print(newBatchFeatureMaps.shape)

    Z = torch.sum(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), dim=1)
    # print(type(Z.shape))
    return Z


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

        featuremap_dict = {}

        # for index in range(len(conv_layers)):
        for index in range(1):
            featuremap_dict[str(index)] = get_feature_maps(model, batch_images)

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
