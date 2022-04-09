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
                #print('Accuracy on {loader_type}: '.format(loader_type=loader_type), accuracy)
            #else:
                #print('Accuracy after ', epoch + 1, ' epochs : ', accuracy)
            return accuracy