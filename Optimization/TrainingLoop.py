import torch
import torch.nn as nn
from ModelEvaluation.Evaluation import Evaluation
from Visualization.LossCurve import LossCurve


class TrainLoop():
    def __init__(self):
        self

    def Tloop(model, epochs, optimizer, learning_rate, train_loader, test_loader, cloned_model=None):
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
            batch_counter = 0
            for images, labels in train_loader:
                model.train()
                # batch_counter += 1
                # print(batch_counter)
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

            # evaluation of model on test data
            if (cloned_model == None):
                Evaluation.Eval(model, epoch, test_loader)
            else:
                cloned_model.load_state_dict(
                    model.state_dict())  # this is recquired so that new weights are tranfered for testing
                Evaluation.Eval(cloned_model, epoch, test_loader)

        # Plotting Loss Curve
        LossCurve.PlotCurve(loss_values, epochs)
