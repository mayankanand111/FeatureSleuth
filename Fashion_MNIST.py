import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split



T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.FashionMNIST("./data/mnist/train_data", train= True, download = True, transform = T)
test_data = torchvision.datasets.FashionMNIST("./data/mnist/test_data", train= False, download = True, transform = T)

print("Train data size - ", len(train_data))
print("Test data size - ", len(test_data))
# Dividing train data again to validation data
# NEW_TRAIN_SIZE = 50000
# VALIDATION_SIZE = 10000
# train_data2, val_data = random_split(train_data, [NEW_TRAIN_SIZE, VALIDATION_SIZE])

# Loading batch of data of size 32 at a time.
NUM_BATCH = 64
train_loader = DataLoader(train_data, batch_size = NUM_BATCH, shuffle=True)
# val_loader = DataLoader(val_data, batch_size = NUM_BATCH)
test_loader = DataLoader(test_data, batch_size = NUM_BATCH, shuffle=True)

# Viewing the data
plt.imshow(train_data[7][0][0], cmap = 'gray')
plt.show()
# print("Target Label : ",train_data[7][1])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 15, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(15, 30, 5)
        self.do2 = nn.Dropout2d(p=0.5)
        self.mp2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        # self.conv2 = nn.Conv2d(30, 60, 5)
        # self.do2 = nn.Dropout2d(p=0.5)
        # self.mp2 = nn.MaxPool2d(2)
        # self.relu2 = nn.ReLU()

        self.fl_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480, 64), )
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout2d(p=0.5)
        self.linear3 = nn.Linear(64, 10)
        self.sm3 = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.mp1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.do2(output)
        output = self.mp2(output)
        output = self.relu2(output)

        output = self.fl_3(output)
        output = self.relu3(output)
        output = self.dp3(output)
        output = self.linear3(output)
        output = self.sm3(output)

        return output

# class ConvNet(nn.Module):
    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #
    #     self.conv1 = nn.Conv2d(1, 15, 5)
    #     self.mp1 = nn.MaxPool2d(2)
    #     self.relu1 = nn.ReLU()
    #
    #     self.conv2 = nn.Conv2d(15, 30, 5)
    #     self.do2 = nn.Dropout2d(p=0.5)
    #     self.mp2 = nn.MaxPool2d(2)
    #     self.relu2 = nn.ReLU()
    #
    #     self.conv3 = nn.Conv2d(30, 30, 3)
    #     # self.do3=nn.Dropout2d(p = 0.5)
    #     # self.mp2=nn.MaxPool2d(2)
    #     self.relu3 = nn.ReLU()
    #
    #     self.fl_3 = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(120, 64), )
    #     self.relu3 = nn.ReLU()
    #     self.dp3 = nn.Dropout2d(p=0.5)
    #     self.linear3 = nn.Linear(64, 10)
    #     self.sm3 = nn.LogSoftmax(dim=1)
    #
    # def forward(self, input):
    #     output = self.conv1(input)
    #     output = self.mp1(output)
    #     output = self.relu1(output)
    #
    #     output = self.conv2(output)
    #     output = self.do2(output)
    #     output = self.mp2(output)
    #     output = self.relu2(output)
    #
    #     output = self.conv3(output)
    #     # output=self.do3(output)
    #     output = self.relu3(output)
    #
    #     output = self.fl_3(output)
    #     output = self.relu3(output)
    #     output = self.dp3(output)
    #     output = self.linear3(output)
    #     output = self.sm3(output)
    #
    #     return output

    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )
    #
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )
    #
    #     self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
    #     self.drop = nn.Dropout2d(0.25)
    #     self.fc2 = nn.Linear(in_features=600, out_features=120)
    #     self.fc3 = nn.Linear(in_features=120, out_features=10)
    #
    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.view(out.size(0), -1)
    #     out = self.fc1(out)
    #     out = self.drop(out)
    #     out = self.fc2(out)
    #     out = self.fc3(out)
    #
    #     return out


    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #
    #     self.convlayer1 = nn.Sequential(
    #         nn.Conv2d(1, 32, 3, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )
    #
    #     self.convlayer2 = nn.Sequential(
    #         nn.Conv2d(32, 64, 3),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )
    #
    #     self.fc1 = nn.Linear(64 * 6 * 6, 600)
    #     self.drop = nn.Dropout2d(0.25)
    #     self.fc2 = nn.Linear(600, 120)
    #     self.fc3 = nn.Linear(120, 10)
    #
    # def forward(self, x):
    #     x = self.convlayer1(x)
    #     x = self.convlayer2(x)
    #     x = x.view(-1, 64 * 6 * 6)
    #     x = self.fc1(x)
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #
    #     return F.log_softmax(x, dim=1)


# defining loss function
# loss_f = nn.CrossEntropyLoss()
loss_f = nn.NLLLoss()

# Defining optimizer
cnn_model = ConvNet()
params = cnn_model.parameters()
optimizer = optim.Adam(params = params, lr = 1e-3)


def validate(model, data):
  total = 0
  correct = 0
  for i, (images, labels) in enumerate(data):
    images = images
    x = model(images)
    value, pred = torch.max(x, 1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)

  return correct  *100 / total


EPOCHS = 10

for epoch in range(EPOCHS):
    losses = []
    accuracies = []
    total = 0
    correct = 0
    for idx, batch in enumerate(train_loader):
        X, y = batch
        cnn_model.train()
        output = cnn_model(X)
        loss = loss_f(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        value, pred = torch.max(output, 1)
        total += output.size(0)
        correct += torch.sum(pred == y)

    cnn_model.eval()
    accuracy = float(validate(cnn_model, test_loader))
    print(f'Epoch {epoch + 1}', end=', ')
    print(f'train loss : {torch.tensor(losses).mean():.2f}', end=', ')
    # print(f'val loss : {losses2} val accuracy : {accuracy}')
    print(f'val accuracy : {accuracy} train accuracy : {correct * 100 / total}')

######### Feature maps


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
              pred_ratio = model(images)
              _, pred_labels = torch.max(pred_ratio, 1)
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



loss_criterion = nn.NLLLoss()
cloned_model=None
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

#selectinf type of optimizer
if optimizer == 'ADAM':
    optimizer = torch.optim.Adam(cnn_model.parameters(),0.001)
elif optimizer == 'SGD':
    optimizer = torch.optim.SGD(cnn_model.parameters(), 0.001)

epochs =10


cnn_model = ConvNet()
model = cnn_model
params = cnn_model.parameters()
optimizer = optim.Adam(params = params, lr = 1e-3)

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

                  Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
                  B_norm = torch.linalg.norm(feature_cachearr, dim=1, keepdim=True)  # Size (1, b).

                  # Distance matrix of size (b, n).
                  cosine_similarity = ((Z @ feature_cachearr.T) / (Z_norm @ B_norm.T)).T

                  cosine_similarity = cosine_similarity.T
                  matched_index = torch.argmax(cosine_similarity, dim=1)



                  # removing image from batch which got sucessfull hit in cache
                  total_indexs = torch.tensor(range(len(images)))
                  index_hitlist = total_indexs[torch.eq(label_cachearr[matched_index] , labels)]

                  #adding cahce hits and sucessful cache hits
                  cache_hits += label_cachearr[matched_index].shape[0]
                  successfull_cache_hits += index_hitlist.shape[0]

                  indexes_toremove = index_hitlist
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
# LossCurve.PlotCurve(loss_values,epochs)
