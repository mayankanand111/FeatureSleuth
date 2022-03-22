import torch

class Evaluation():
    def __init__(self):
        self

    def Eval(model,epoch,test_loader):
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
            print('Accuracy after ', epoch + 1, ' epochs : ', accuracy)