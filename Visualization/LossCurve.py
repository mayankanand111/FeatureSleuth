import matplotlib.pyplot as plt


class LossCurve:
    def __init__(self):
        self

    def PlotCurve(loss_values,epochs):
        plt.plot(range(epochs), loss_values, 'blue')
        plt.title('Loss decay')
        plt.xlabel('number of epochs')
        plt.ylabel('Loss')
        plt.show()