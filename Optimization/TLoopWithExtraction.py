import numpy as np
import torch
import torch.nn as nn
from ModelEvaluation.Evaluation import Evaluation
from Visualization.LossCurve import LossCurve
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

class TLoopWithExtraction():
    def __init__(self):
        self

    def Tloop_Extraction(model,epochs,optimizer,learning_rate,train_loader,test_loader,cloned_model=None):
        #loss criteria
        loss_criterion = nn.NLLLoss()

        #selectinf type of optimizer
        if optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(),learning_rate)
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
            for images, labels in train_loader:
                model.train()
                batch_counter +=1
                # if feature cache is not empty than do cosine similarity check
                if(feature_cachearr!=None):
                    print("Current Batch is {} and fmap count is {}".format(batch_counter,len(feature_cachearr)))
                    #first get feature maps for new batch from model trained weights till now
                    with torch.no_grad():
                        feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
                        newBatchFeatureMaps = feature_extractor(images)
                        # now we have feature maps of new batch and feature maps cache so far to perform cosine similarity check
                        index_hitlist = []
                        # for index in range(len(images)):
                        #     matched_index = torch.argmax(F.cosine_similarity(
                        #     torch.sum(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3),dim=1)[index],
                        #     feature_cachearr, dim=1))
                        #     if(label_cachearr[matched_index].item()==labels[index].item()):
                        #         #remove that image from batch
                        #         successfull_cache_hits += 1
                        #         cache_hits += 1
                        #         index_hitlist.append(index)
                        #     else:
                        #         cache_hits += 1


                        ################### Akhilesh

                        Z = torch.sum(newBatchFeatureMaps['conv1'].flatten(start_dim=2, end_dim=3), dim=1)

                        Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
                        B_norm = torch.linalg.norm(feature_cachearr, dim=1, keepdim=True)  # Size (1, b).

                        # Distance matrix of size (b, n).
                        cosine_similarity = ((Z @ feature_cachearr.T) / (Z_norm @ B_norm.T)).T

                        cosine_similarity = cosine_similarity.T
                        matched_index = torch.argmax(cosine_similarity, dim=1)


                        index_hitlist = label_cachearr[matched_index] [
                            torch.eq(label_cachearr[matched_index] , labels)]

                        ##################

                        #removing image from batch which got sucessfull hit in cache
                        total_indexs = torch.tensor(range(len(images)))
                        indexes_toremove = torch.tensor(index_hitlist)
                        out, c = torch.cat([total_indexs, indexes_toremove]).unique(return_counts=True)
                        indexes_tokeep = out[c == 1]
                        images = torch.index_select(images, 0, indexes_tokeep)
                        labels = torch.index_select(labels, 0, indexes_tokeep)

                # forward pass
                outputs = model(images)
                # finding loss
                loss = loss_criterion(outputs, labels)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    #extract feature maps fo the batch and stack it to cache stack
                    feature_extractor = create_feature_extractor(model, return_nodes=['conv1'])
                    out = feature_extractor(images)
                    feature_cache.append(torch.sum(out['conv1'].flatten(start_dim=2, end_dim=3), 1))
                    label_cache.append(labels)
                    # stacking tensor to concatenate
                    feature_cachearr = torch.cat(feature_cache)
                    # feature_cachearr = torch.reshape(feature_cachearr, (
                    # feature_cachearr.size()[0] * feature_cachearr.size()[1], feature_cachearr.size()[2]))
                    label_cachearr = torch.cat(label_cache).flatten()
                running_loss += loss.item()
            # print("Total cache hits in {} epoch are : {}".format(epoch,cache_hits))
            # print("Total successful cache hits in {} epoch are : {}".format(epoch, successfull_cache_hits))
            loss_values.append(running_loss / len(train_loader))

            #evaluation of model on test data
            if (cloned_model == None):
                Evaluation.Eval(model,epoch,test_loader)
            else:
                cloned_model.load_state_dict(model.state_dict()) # this is recquired so that new weights are tranfered for testing
                Evaluation.Eval(cloned_model, epoch, test_loader)

        # Plotting Loss Curve
        LossCurve.PlotCurve(loss_values,epochs)


