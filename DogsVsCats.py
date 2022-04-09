import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# from sklearn.metrics import classification_report, confusion_matrix



BATCH_SIZE = 500
EPOCHS = 20
LEARNING_RATE = 0.0001


train_dir = '/content/drive/MyDrive/ML_Project/kagglecatsanddogs_3367a/PetImages'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_folder = ImageFolder(root=train_dir,transform=transform)



train_loader2 = DataLoader(train_folder,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

train_loader1, valid_loader1 = torch.utils.data.random_split(train_loader2.dataset, (12500, 12506))

evens = list(range(0, len(train_loader1), 2))
# odds = list(range(1, len(trainset), 2))
trainset_1 = torch.utils.data.Subset(train_loader1, evens)
odds = list(range(1, len(valid_loader1), 5))
valset_1 = torch.utils.data.Subset(valid_loader1, odds)

# train_loader1, valid_loader1 = torch.utils.data.random_split(train_loader2.dataset, (20000, 5008))
train_loader = DataLoader(trainset_1,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
valid_loader = DataLoader(valset_1,
                          batch_size=BATCH_SIZE,
                          shuffle=True)


class CatAndDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        #         print(X.shape)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        #         X = torch.sigmoid(X)
        return X

import torch

if torch.cuda.is_available():
      device = torch.device("cuda")
else:
      device = torch.device("cpu")


def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images,labels = images.to(device),labels.to(device)
        x = model(images)
        value, pred = torch.max(x, 1)
        value, pred = value.to(device), pred.to(device)
        # pred = pred.c
        total += x.size(0)
        correct += torch.sum(pred == labels)

    return correct  *100 / total

# net = Net()
cnn_model = CatAndDogNet().to(device)
# cnn_model.to(device)
params = cnn_model.parameters()
optimizer = optim.Adam(params = params, lr = 1e-3)

# loss_f = nn.NLLLoss()
loss_f  = nn.CrossEntropyLoss()

from tqdm import tqdm


# Training the network
EPOCHS = 5

for epoch in tqdm(range(EPOCHS)):
    print(epoch)
    losses = []
    accuracies = []
    total = 0
    correct = 0
    for idx, batch in enumerate(train_loader):
        # batch.to(device)

        X, y = batch
        print(X.shape)
        X, y = X.to(device), y.to(device)
        # X = X.cuda()
        # y = y.cuda()
        cnn_model.train()
        output = cnn_model(X)
        loss = loss_f(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        value, pred = torch.max(output, 1)
        value, pred = value.to(device), pred.to(device)
        total += output.size(0)
        correct += torch.sum(pred == y)

    cnn_model.eval()
    accuracy = float(validate(cnn_model, valid_loader))
    print(f'Epoch {epoch + 1}', end=', ')
    print(f'train loss : {torch.tensor(losses).mean():.2f}', end=', ')
    # print(f'val loss : {losses2} val accuracy : {accuracy}')
    print(f'val accuracy : {accuracy} train accuracy : {correct * 100 / total}')


# net = Net()
cnn_model = CatAndDogNet().to(device)
# cnn_model.to(device)
params = cnn_model.parameters()
optimizer = optim.Adam(params = params, lr = 1e-3)

# loss_f = nn.NLLLoss()
loss_f  = nn.CrossEntropyLoss()

loss_criterion = nn.CrossEntropyLoss()

import torch

class Evaluation():
  def __init__(self):
      self

  def Eval(model,epoch,test_loader,is_only_test=False):
      with torch.no_grad():
          model.eval()
          correct_samples = 0
          total_samples = 0
          for images, labels in test_loader:
              images, labels = images.to(device), labels.to(device)
              pred_ratio = model(images)
              _, pred_labels = torch.max(pred_ratio, 1)
              pred_labels = pred_labels.to(device)
              total_samples += labels.size(0)
              correct_samples += (pred_labels == labels).sum().item()
          accuracy = (correct_samples / total_samples) * 100
          loader_type = ""
          if(is_only_test):
              if(total_samples >10000):
                  loader_type = "train set"
              else:
                  loader_type = "test set"
              print('Accuracy on {loader_type}: '.format(loader_type=loader_type), accuracy)
          else:
              print('Accuracy after ', epoch + 1, ' epochs : ', accuracy)

# %%time
import time
torch.set_num_threads(1)


cloned_model=None
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

#selectinf type of optimizer
if optimizer == 'ADAM':
    optimizer = torch.optim.Adam(cnn_model.parameters(),0.001)
elif optimizer == 'SGD':
    optimizer = torch.optim.SGD(cnn_model.parameters(), 0.001)

epochs = 5
model = cnn_model
loss_values = []
for epoch in tqdm(range(epochs)):
    start = time.time()
    running_loss = 0
    feature_cache = []
    label_cache = []
    feature_cachearr = None
    label_cachearr = None
    cache_hits = 0
    successfull_cache_hits = 0
    batch_counter = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        model.train()
        batch_counter +=1
        if(batch_counter>len(train_loader)/2):
          # if feature cache is not empty than do cosine similarity check
          if(feature_cachearr!=None):
              # print("Current Batch is {} and fmap count is {}".format(batch_counter,len(feature_cachearr)))
              #first get feature maps for new batch from model trained weights till now
              with torch.no_grad():
                  feature_extractor = create_feature_extractor(model, return_nodes=['conv2'])
                  newBatchFeatureMaps = feature_extractor(images)
                  # now we have feature maps of new batch and feature maps cache so far to perform cosine similarity check
                  index_hitlist = []


                  ################### Akhilesh

                  Z = torch.sum(newBatchFeatureMaps['conv2'].flatten(start_dim=2, end_dim=3), dim=1)
                  Z = Z.to(device)
                  Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
                  B_norm = torch.linalg.norm(feature_cachearr, dim=1, keepdim=True)  # Size (1, b).
                  Z_norm , B_norm = Z_norm.to(device) , B_norm.to(device)
                  # Distance matrix of size (b, n).
                  cosine_similarity = ((Z @ feature_cachearr.T) / (Z_norm @ B_norm.T)).T

                  cosine_similarity = cosine_similarity.T
                  matched_index = torch.argmax(cosine_similarity, dim=1)
                  matched_index = matched_index.to(device)
                  #TODO: below code has to be reoved as right index fix is implemented
                  # below code gets wrong indexes in batch to remove image from batch fix is done below total indexes
                  # index_hitlist2 = label_cachearr[matched_index2] [
                  #     torch.eq(label_cachearr[matched_index2] , labels)]


                  # removing image from batch which got sucessfull hit in cache
                  total_indexs = torch.tensor(range(len(images)))
                  total_indexs = total_indexs.to(device)
                  index_hitlist = total_indexs[torch.eq(label_cachearr[matched_index] , labels)]
                  index_hitlist = index_hitlist.to(device)
                  #adding cahce hits and sucessful cache hits
                  cache_hits += label_cachearr[matched_index].shape[0]
                  successfull_cache_hits += index_hitlist.shape[0]

                  indexes_toremove = index_hitlist
                  start1 = time.time()
                  out, c = torch.cat([total_indexs, indexes_toremove]).unique(return_counts=True)
                  out, c = out.to(device), c.to(device)
                  end1 = time.time()
                  # print("First concat operation",( end1 - start1)/60)
                  indexes_tokeep = out[c == 1]
                  images = torch.index_select(images, 0, indexes_tokeep)
                  labels = torch.index_select(labels, 0, indexes_tokeep)
                  images, labels = images.to(device), labels.to(device)

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
            feature_extractor = create_feature_extractor(model, return_nodes=['conv2'])
            out = feature_extractor(images)
            feature_cache.append(torch.sum(out['conv2'].flatten(start_dim=2, end_dim=3), 1))
            label_cache.append(labels)
            # stacking tensor to concatenate
            start2 = time.time()
            feature_cachearr = torch.cat(feature_cache)
            feature_cachearr = feature_cachearr.to(device)
            end2 = time.time()
            # print("Second concat operation",( end2 - start2)/60)
            # feature_cachearr = torch.reshape(feature_cachearr, (
            # feature_cachearr.size()[0] * feature_cachearr.size()[1], feature_cachearr.size()[2]))
            start3 = time.time()
            label_cachearr = torch.cat(label_cache).flatten()
            label_cachearr = label_cachearr.to(device)
            end3 = time.time()
            # print("Third concat operation",( end3 - start3)/60)

        running_loss += loss.item()

    print("Total cache hits in {} epoch are : {}".format(epoch,cache_hits))
    print("Total successful cache hits in {} epoch are : {}".format(epoch, successfull_cache_hits))
    loss_values.append(running_loss / len(train_loader))

    #evaluation of model on test data
    if (cloned_model == None):
        Evaluation.Eval(model,epoch,valid_loader)
    else:
        cloned_model.load_state_dict(model.state_dict()) # this is recquired so that new weights are tranfered for testing
        Evaluation.Eval(cloned_model, epoch, valid_loader)

# Plotting Loss Curve
# LossCurve.PlotCurve(loss_values,epochs)
# print("hello")
    end = time.time()
    print("Epoch -- ",epoch,(end - start)/60)
# A few seconds later
