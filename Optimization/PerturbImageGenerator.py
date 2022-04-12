import torch
import torch.nn as nn

from DataLoader.Loader import TensorDataset


class PerturbImageGenerator:
    def __init__(self):
        self

    def get(self, model, train_loader, epsilon=0.05):
        # loss criteria
        loss_criterion = nn.NLLLoss()
        tensor_dataset = TensorDataset(None, None)
        loss_values = []
        input_grad = None
        for images, labels in train_loader:
            images.requires_grad = True
            # model.train()
            # forward pass
            outputs = model(images)
            # finding loss
            loss = loss_criterion(outputs, labels)
            # backward pass
            loss.backward()

            product_value = torch.mul(epsilon, torch.sign(images.grad))
            images_new = torch.clone(images)
            perturbed_images = images_new.add(product_value)
            tensor_dataset.append(perturbed_images, labels)

        return tensor_dataset


