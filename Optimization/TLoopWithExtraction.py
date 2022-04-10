import numpy as np
import torch
import torch.nn as nn
from ModelEvaluation.Evaluation import Evaluation
from Visualization.LossCurve import LossCurve
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from tqdm import tqdm


class TLoopWithExtraction():
    def __init__(self):
        self

    def getfeatureVector(out, labels, count_dic, vect_dic):
        std = torch.std(out['conv1'].flatten(start_dim=2, end_dim=3), 2)
        mean = torch.abs(torch.mean(out['conv1'].flatten(start_dim=2, end_dim=3), 2))
        vector = torch.stack((mean, std), dim=2).flatten(start_dim=1, end_dim=2)
        for i in range(10):
            total_items = vector[torch.where(labels == i)].shape[0]
            vect_dic[i] = ((vect_dic[i].reshape(1, 30) * count_dic[i]) + torch.sum(vector[torch.where(labels == i)], dim=0, keepdim=True)) / (count_dic[i] + total_items)
            count_dic[i] += total_items
        return vect_dic, count_dic

    def getcurrentbatchVectors(newBatchFeatureMaps):
        std = torch.std(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), 2)
        mean = torch.abs(torch.mean(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), 2))
        vector = torch.stack((mean, std), dim=2).flatten(start_dim=1, end_dim=2)
        return vector

    def Tloop_Extraction(model, epochs, optimizer, learning_rate, train_loader, test_loader, cloned_model=None):
        # loss criteria
        loss_criterion = nn.NLLLoss()

        # selectinf type of optimizer
        if optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), learning_rate)

        loss_values = []
        for epoch in range(epochs):
            running_loss = 0
            feature_cache = []
            label_cache = []
            feature_cachearr = None
            label_cachearr = None
            cache_hits = 0
            successfull_cache_hits = 0
            batch_counter = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            count_dictionary = torch.zeros((10,1))   # this np array is used to store count of each class images that we got so far
            vector_dictionary = torch.zeros((10,30)) # this is the vector cache that is used to store the latest cahced object for each class
            for batch_idx, (images, labels) in loop:
                model.train()
                # if feature cache is not empty than do cosine similarity check
                if (count_dictionary[0].item() != 0):
                    # print("Current Batch is {} and fmap count is {}".format(batch_counter,len(feature_cachearr)))
                    # first get feature maps for new batch from model trained weights till now
                        with torch.no_grad():
                            feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
                            newBatchFeatureMaps = feature_extractor(images)
                            newBatchVectors = TLoopWithExtraction.getcurrentbatchVectors(newBatchFeatureMaps)
                            # now we have feature maps of new batch and feature maps cache so far to perform cosine similarity check
                            Z_norm = torch.linalg.norm(newBatchVectors, dim=1, keepdim=True)  # Size (n, 1).
                            B_norm = torch.linalg.norm(vector_dictionary, dim=1, keepdim=True)  # Size (1, b).

                            # Distance matrix of size (b, n).
                            cosine_similarity = ((newBatchVectors @ vector_dictionary.T) / (Z_norm @ B_norm.T)).T

                            cosine_similarity = cosine_similarity.T
                            predicted_labels = torch.argmax(cosine_similarity, dim=1)

                            # removing image from batch which got sucessfull hit in cache
                            total_indexs = torch.tensor(range(len(images)))
                            index_hitlist = total_indexs[torch.eq(predicted_labels, labels)]

                            # adding cahce hits and sucessful cache hits
                            cache_hits += predicted_labels.shape[0]
                            successfull_cache_hits += index_hitlist.shape[0]

                            indexes_toremove = index_hitlist
                            out, c = torch.cat([total_indexs, indexes_toremove]).unique(return_counts=True)
                            indexes_tokeep = out[c == 1]
                            images = torch.index_select(images, 0, indexes_tokeep)
                            labels = torch.index_select(labels, 0, indexes_tokeep)

                            # # clearing feature cache after each batch
                            # feature_cachearr = []
                            # label_cachearr = []

                # forward pass
                outputs = model(images)
                # finding loss
                loss = loss_criterion(outputs, labels)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # extract feature maps of the batch and stack it to cache stack
                    feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
                    out = feature_extractor(images)
                    vector_dictionary,count_dictionary = TLoopWithExtraction.getfeatureVector(out,labels,count_dictionary,vector_dictionary)
                    # feature_cachearr =  torch.sum(out['conv1'].flatten(start_dim=2, end_dim=3), 1)
                    # label_cachearr = labels
                    # stacking tensor to concatenate
                    # feature_cachearr = torch.cat(feature_cache)
                    # feature_cachearr = torch.reshape(feature_cachearr, (
                    # feature_cachearr.size()[0] * feature_cachearr.size()[1], feature_cachearr.size()[2]))
                    # label_cachearr = torch.cat(label_cache).flatten()
                running_loss += loss.item()
                # evaluation of model on test data
                train_accuracy = 0
                test_accuracy = 0
                if (cloned_model == None):
                    test_accuracy = Evaluation.Eval(model, epoch, test_loader)
                else:
                    cloned_model.load_state_dict(
                        model.state_dict())  # this is recquired so that new weights are tranfered for testing
                    test_accuracy = Evaluation.Eval(cloned_model, epoch, test_loader)

                loop.set_description(f"Epoch[{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item(), accuracy=test_accuracy, running_loss=running_loss,
                                 cache_hits=cache_hits, successful_cache_hits=successfull_cache_hits)
            # print("Total cache hits in {} epoch are : {}".format(epoch,cache_hits))
            # print("Total successful cache hits in {} epoch are : {}".format(epoch, successfull_cache_hits))
            loss_values.append(running_loss / len(train_loader))
            if (epoch + 1 == epochs):
                if (cloned_model == None):
                    train_accuracy = Evaluation.Eval(model, epoch, train_loader)
                    print("Accuracy on train set: {}".format(train_accuracy))
                    test_accuracy =  Evaluation.Eval(model, epoch, test_loader)
                    print("Accuracy on test set: {}".format(test_accuracy))

                else:
                    cloned_model.load_state_dict(
                        model.state_dict())  # this is recquired so that new weights are tranfered for testing
                    train_accuracy = Evaluation.Eval(cloned_model, epoch, train_loader)
                    print("Accuracy on train set {}:".format(train_accuracy))
                    test_accuracy = Evaluation.Eval(cloned_model, epoch, test_loader)
                    print("Accuracy on test set: {}".format(test_accuracy))
        # Plotting Loss Curve
        LossCurve.PlotCurve(loss_values, epochs)
