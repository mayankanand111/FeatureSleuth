
from Research.DataAnalysis import LoadData

if __name__ == '__main__':
    train_data = LoadData.Load_Data()
    LoadData.PrintImage(train_data)