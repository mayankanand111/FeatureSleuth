import torch
import torch.nn as nn
from ModelEvaluation.Evaluation import Evaluation
from Visualization.LossCurve import LossCurve


class TrainLoop():
    def __init__(self):
        self

    def Tloop(model,epochs,optimizer,learning_rate,train_loader,test_loader):
        #loss criteria
        loss_criterion = nn.NLLLoss()

        #selectinf type of optimizer
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),learning_rate)
        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), learning_rate)

        loss_values = []
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in train_loader:
                model.train()
                # forward pass
                outputs = model(images)
                # finding loss
                loss = loss_criterion(outputs, labels)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            loss_values.append(running_loss / len(train_loader))

            #evaluation of model on test data
            Evaluation.Eval(model,epoch,test_loader)

        # Plotting Loss Curve
        LossCurve.PlotCurve(loss_values,epochs)

