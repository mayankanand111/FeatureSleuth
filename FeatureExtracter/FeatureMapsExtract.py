import numpy
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
            for f in range(feature_map.size()[1]):
                processed.append(feature_map[0][f])

        feature_map_image = []
        for i in range(len(processed)):
            feature_map_image.append(processed[i])

        return feature_map_image

    def getfeatures_from_loader(input_loader,model):
        train_feature_maps = []
        train_labels = []
        for images, labels in input_loader:
            ExtractedFmaplist = FmapExtract.extract_featuremaps(images,model)
            for fmap in range(len(ExtractedFmaplist)):
                train_feature_maps.append(ExtractedFmaplist[fmap].detach().reshape(1, ExtractedFmaplist[fmap].size()[0], ExtractedFmaplist[fmap].size()[1]))
                train_labels.append(labels.item())
        # feature_list = []
        # for i in train_feature_maps:
        #     feature_list.append(torch.tensor(i.reshape(1, 24, 24)))
        return np.array(train_feature_maps), np.array(train_labels, dtype=np.uint8)

    def extract_featuremaps_new(batch_images,model):
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

