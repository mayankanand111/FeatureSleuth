import torchvision.transforms
from numpy.linalg import norm
from Models.model import BaseModel
from torch import nn
import torch
from DataLoader import Loader
import numpy as np
import matplotlib.pyplot as plt
from Conf import DataConfig
from Conf.DataConfig import MNISTConfig
import hydra
from hydra.core.config_store import ConfigStore


@hydra.main(config_path="Conf", config_name="DataConfig")
def train_half_dataset(cfg: MNISTConfig) -> None:
    # we will save the conv layer weights in this list
    model_weights = []
    # we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    cnn_model = BaseModel()
    model_children = list(cnn_model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective wights to the list
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
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn_model.to(device)

    # first train on half of the images
    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                          cfg.hyperparams.batch_size)


    test_loader = Loader.Test_Loader.load_test_dataset(cfg.params.test_data_path, cfg.params.test_labels_path,
                                                       cfg.hyperparams.batch_size)

    EPOCHS = 1
    for epoch in range(EPOCHS):
        for idx, batch in enumerate(test_loader):
            X, y = batch
            break

    # image = test_data[0][0].reshape(test_data[0][0].shape[1], -1)
    image = X[0][0]
    # plt.imshow(image)
    # plt.show()

    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    image = T(image.numpy())
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    feature_maps_stores = []
    # feature_maps_stores_array1 = np.empty((32, 15, 24, 24))
    feature_maps_stores_array1 = []
    # feature_maps_stores_array2 = np.empty((32, 30, 20, 20))
    feature_maps_stores_array2 = []
    labels_stores = []
    labels_array = np.empty((1,))

    train_loader = Loader.Train_Loader.load_train_dataset(cfg.params.train_data_path, cfg.params.train_labels_path,
                                                          cfg.hyperparams.batch_size)

    for img, label in train_loader:
        # print(len(label))
        # print(img.shape)
        # img = img.numpy()
        # print(f"Image shape before: {img.shape}")
        # img = img.unsqueeze(0)
        # print(f"Image shape after: {img.shape}")
        img = img.to(device)
        outputs = []
        names = []
        for layer in conv_layers[0:]:
            img = layer(img)
            outputs.append(img)
            names.append(str(layer))

        processed1 = []
        # processed1 = np.empty((32, 15, 24, 24))
        processed2 = []
        # processed2 = np.empty((32, 30, 20, 20))

        for idx, feature_map in enumerate(outputs):
            feature_map = feature_map.squeeze(0)

            # gray_scale = torch.sum(feature_map,0)
            # print(gray_scale.shape)
            gray_scale = feature_map / feature_map.shape[0]
            # print(gray_scale.data.numpy().shape)
            if (idx == 0):
                # processed1 = np.append(processed1,gray_scale.data.numpy())
                processed1.append(gray_scale.data.numpy())
            else:
                # processed2 = np.append(processed2,gray_scale.data.numpy())
                processed2.append(gray_scale.data.numpy())

        # feature_maps_stores.extend(processed)
        # feature_maps_stores_array1 = np.append(feature_maps_stores_array1,processed1)
        feature_maps_stores_array1.extend(processed1)
        # feature_maps_stores_array2 = np.append(feature_maps_stores_array2, processed2)
        feature_maps_stores_array2.extend(processed2)

        # array = np.append(array, np.array([[1,3,5]]), axis=0)
        # array = np.append(array, np.array([[2,4,6]]), axis=0)

        labels_stores.extend(label)

    K = 2
    labels_stores_new = [ele for ele in labels_stores for i in range(K)]
    # print(len(outputs))
    # print feature_maps
    # for feature_map in outputs:
    # print(feature_map.shape)

    feature_maps_stores11 = np.concatenate(feature_maps_stores_array1, axis=0)
    feature_maps_stores22 = np.concatenate(feature_maps_stores_array2, axis=0)

    feature_maps_stores11 = np.concatenate(feature_maps_stores11, axis=0)
    feature_maps_stores22 = np.concatenate(feature_maps_stores22, axis=0)

    labels_stores_new1 = labels_stores_new[::2]
    labels_stores_new2 = labels_stores_new[1::2]

    K = 15
    res1 = [ele for ele in labels_stores_new1 for i in range(K)]

    K = 30
    res2 = [ele for ele in labels_stores_new2 for i in range(K)]

    print("processed[0] : ", processed[0].shape)
    # plt.imshow(processed[0])
    # plt.show()
    print("Printed processed image")

    print(feature_maps_stores11[0].shape)
    emp_dict = {"cos_sims": [], "sim_images": [], "sim_labels": []}

    for i, j in zip(feature_maps_stores11, res1):
        new_img = processed[0].flatten()
        # print(type(i))
        # print(type(new_img))
        cos_sim = np.dot(new_img, i.flatten()) / (norm(new_img) * norm(i.flatten()))
        # print("cos_sim",cos_sim)
        if (cos_sim >= 0.6 and cos_sim < 1):
            # print("cos_sim : ", cos_sim)
            emp_dict["cos_sims"].append(cos_sim)
            emp_dict['sim_images'].append(i)
            emp_dict['sim_labels'].append(j)

            # plt.title(j)
            # plt.imshow(i)
            # plt.show()
            # print("True")

    print(sorted(emp_dict['cos_sims']))
    max_sim = sorted(emp_dict['cos_sims'])[-1]
    idx_max_sim = emp_dict['cos_sims'].index(max_sim)
    plt.title(emp_dict['sim_labels'][idx_max_sim])
    plt.imshow(emp_dict['sim_images'][idx_max_sim])
    plt.show()

    print("program end")


if __name__ == "__main__":
    main()
