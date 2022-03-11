from Research.DataAnalysis import LoadData
from Research.DataAnalysis import LoadLabels

if __name__ == '__main__':
    train_data = LoadData.Load_Data()
    LoadData.PrintImage(train_data)
    load_labels = LoadLabels()
    train_labels = load_labels.load_labels()
    train_label_count = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "0": 0}

    for i in range(len(train_labels)):
        count = train_label_count["{}".format(train_labels[i])]
        train_label_count["{}".format(train_labels[i])] = count + 1

    print(train_label_count)