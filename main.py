import numpy as np
from PIL import Image

from Research.DataAnalysis import LoadData
from Research.DataAnalysis import LoadLabels


def take_average(dataset, labels, label_count_dict):
    number_dict = {"1": np.zeros_like(dataset.shape[1]),
                   "2": np.zeros_like(dataset.shape[1]),
                   "3": np.zeros_like(dataset.shape[1]),
                   "4": np.zeros_like(dataset.shape[1]),
                   "5": np.zeros_like(dataset.shape[1]),
                   "6": np.zeros_like(dataset.shape[1]),
                   "7": np.zeros_like(dataset.shape[1]),
                   "8": np.zeros_like(dataset.shape[1]),
                   "9": np.zeros_like(dataset.shape[1]),
                   "0": np.zeros_like(dataset.shape[1])}
    for index in range(len(labels)):
        current_label = "{}".format(labels[index])
        number_dict[current_label] = sum_arrays(number_dict[current_label], dataset[index])

    for k in number_dict:
        number_dict[k] = number_dict[k] / label_count_dict[k]
    return number_dict


def sum_arrays(array1, array2):
    return array1 + array2


def find_possible_outliers(train_dataset, t_labels, average_dict):
    labels = list(average_dict.keys())
    outliers_dict = {}
    # initialize empty array for each label
    for i in range(len(labels)):
        outliers_dict[labels[i]] = {}
    # for i in range(len(t_labels)):
    # return cosine_similarity(train_dataset, average_dict, outliers_dict, t_labels)
    return euclidean_distance(train_dataset, average_dict, outliers_dict, t_labels)


def cosine_similarity(train_dataset, average_dict, outliers_dict, t_labels):
    for i in range(len(t_labels)):
        label = "{}".format(t_labels[i])
        value = calculate_cosine(train_dataset[i], average_dict[label])
        # print("Similarity value: ", value)
        if value < 0.4:
            outliers_dict[label].append(train_dataset[i])

    return outliers_dict


def calculate_cosine(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def euclidean_distance(train_dataset, average_dict, outliers_dict, t_labels):
    for i in range(len(t_labels)):
        label = "{}".format(t_labels[i])
        value = calculate_euclidean(train_dataset[i], average_dict[label])
        # print("Similarity value: ", value)
        outliers_dict = update_outliers_dict(outliers_dict, label, value, train_dataset[i])
    return outliers_dict


def update_outliers_dict(outliers_dict, label, distance, current_image):
    outlier = outliers_dict[label]
    if len(outlier.keys()) < 5:
        outlier[distance] = current_image
    else:
        k = find_least_distance_key(list(outlier.keys()))
        del outlier[k]
        outlier[distance] = current_image

    outliers_dict[label] = outlier
    return outliers_dict


def find_least_distance_key(keys):
    temp = keys[0]
    for i in range(1, len(keys)):
        if temp > keys[i]:
            temp = keys[i]

    return temp


def calculate_euclidean(a1, a2):
    return np.sqrt(np.sum((a1 - a2) ** 2))


if __name__ == '__main__':
    train_data = LoadData.Load_Data()
    # LoadData.PrintImage(train_data[1])
    # print(train_data.shape)
    load_labels = LoadLabels()
    train_labels = load_labels.load_labels()
    train_label_count = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "0": 0}

    for i in range(len(train_labels)):
        count = train_label_count["{}".format(train_labels[i])]
        train_label_count["{}".format(train_labels[i])] = count + 1

    print(train_label_count)
    avg_dict = take_average(train_data, train_labels, train_label_count)
    outlier_dict = find_possible_outliers(train_data, train_labels, avg_dict)
    # for key in avg_dict:
    #     LoadData.PrintImage(avg_dict[key])
    for key in outlier_dict:
        # print(f'Outliers of {key} are {len(outlier_dict[key])}')
        current_outliers = outlier_dict[key]
        final = Image.new('RGB', (28 * 5, 28))
        index_w = 0
        index_h = 0
        for k in current_outliers:
            image = LoadData.PrintImage(current_outliers[k])
            final.paste(image, (index_w, index_h))
            index_w = index_w + image.width
        final.show()
