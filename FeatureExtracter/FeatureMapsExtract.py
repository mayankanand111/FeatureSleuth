import torch
import torch.nn as nn
import numpy as np

class FmapExtract:

    def extract_featuremaps(batch_images,model):
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
        for layer in conv_layers[0:1]:
            image = layer(batch_images)
            outputs.append(image)
            names.append(str(layer))

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(1)
            gray_scale = torch.sum(feature_map, 1)
            gray_scale = gray_scale / feature_map.shape[1]
            processed.append(gray_scale.data.cpu())

        feature_map_image = []
        for i in range(len(processed)):
            feature_map_image = processed[i]

        return feature_map_image

    def getfeatures_from_loader(input_loader,model):
        train_feature_maps = []
        train_labels = []
        for images, labels in input_loader:
            train_labels.append(labels.item())
            train_feature_maps.append(FmapExtract.extract_featuremaps(images,model))
        feature_list = []
        for i in train_feature_maps:
            feature_list.append(torch.tensor(i.reshape(1, 24, 24)))
        return np.array(feature_list), np.array(train_labels, dtype=np.uint8)