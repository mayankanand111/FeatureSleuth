import torch
import torch.nn as nn

from DataLoader.Loader import TensorDataset


class PerturbImageGenerator:
    def __init__(self):
        self

    def get(self, model, train_loader, epsilon=0.05):
        # loss criteria
        loss_criterion = nn.NLLLoss()
        input_grad = None
        original_input = None
        p_original_label = None
        tensor_dataset = TensorDataset(None, None)
        for images, labels in train_loader:
            images.requires_grad = True
            # model.train()
            # forward pass
            outputs = model(images)
            # finding loss
            loss = loss_criterion(outputs, labels)
            model.zero_grad()
            # backward pass
            loss.backward()
            if original_input == None:
                original_input = images
            else:
                original_input = torch.cat((original_input, images))

            if input_grad == None:
                input_grad = images.grad
            else:
                input_grad = torch.cat((input_grad, images.grad))

            if p_original_label == None:
                p_original_label = labels
            else:
                p_original_label = torch.cat((p_original_label, labels))


            # product_value = torch.mul(epsilon, torch.sign(images.grad.data))
            # # images_new = torch.clone(images)
            # perturbed_images = images.add(product_value)
            # tensor_dataset.append(perturbed_images, labels)

        return original_input, input_grad, p_original_label


