
# moved below code from main to Research Folder -> ExplonatoryAnalysis findings
# import numpy as np
# from PIL import Image
#
# from Research.DataAnalysis import LoadData
# from Research.DataAnalysis import LoadLabels
#
#
# def take_average(dataset, labels, label_count_dict):
#     number_dict = {"1": np.zeros_like(dataset.shape[1]),
#                    "2": np.zeros_like(dataset.shape[1]),
#                    "3": np.zeros_like(dataset.shape[1]),
#                    "4": np.zeros_like(dataset.shape[1]),
#                    "5": np.zeros_like(dataset.shape[1]),
#                    "6": np.zeros_like(dataset.shape[1]),
#                    "7": np.zeros_like(dataset.shape[1]),
#                    "8": np.zeros_like(dataset.shape[1]),
#                    "9": np.zeros_like(dataset.shape[1]),
#                    "0": np.zeros_like(dataset.shape[1])}
#     for index in range(len(labels)):
#         current_label = "{}".format(labels[index])
#         number_dict[current_label] = sum_arrays(number_dict[current_label], dataset[index])
#
#     for k in number_dict:
#         number_dict[k] = number_dict[k] / label_count_dict[k]
#     return number_dict
#
#
# def sum_arrays(array1, array2):
#     return array1 + array2
#
#
# def find_possible_outliers(train_dataset, t_labels, average_dict):
#     labels = list(average_dict.keys())
#     outliers_dict = {}
#     # initialize empty array for each label
#     for i in range(len(labels)):
#         outliers_dict[labels[i]] = {}
#     # for i in range(len(t_labels)):
#     # return cosine_similarity(train_dataset, average_dict, outliers_dict, t_labels)
#     return euclidean_distance(train_dataset, average_dict, outliers_dict, t_labels)
#
#
# def cosine_similarity(train_dataset, average_dict, outliers_dict, t_labels):
#     for i in range(len(t_labels)):
#         label = "{}".format(t_labels[i])
#         value = calculate_cosine(train_dataset[i], average_dict[label])
#         # print("Similarity value: ", value)
#         if value < 0.4:
#             outliers_dict[label].append(train_dataset[i])
#
#     return outliers_dict
#
#
# def calculate_cosine(a1, a2):
#     return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
#
#
# def euclidean_distance(train_dataset, average_dict, outliers_dict, t_labels):
#     for i in range(len(t_labels)):
#         label = "{}".format(t_labels[i])
#         value = calculate_euclidean(train_dataset[i], average_dict[label])
#         # print("Similarity value: ", value)
#         outliers_dict = update_outliers_dict(outliers_dict, label, value, train_dataset[i])
#     return outliers_dict
#
#
# def update_outliers_dict(outliers_dict, label, distance, current_image):
#     outlier = outliers_dict[label]
#     if len(outlier.keys()) < 5:
#         outlier[distance] = current_image
#     else:
#         k = find_least_distance_key(list(outlier.keys()))
#         del outlier[k]
#         outlier[distance] = current_image
#
#     outliers_dict[label] = outlier
#     return outliers_dict
#
#
# def find_least_distance_key(keys):
#     temp = keys[0]
#     for i in range(1, len(keys)):
#         if temp > keys[i]:
#             temp = keys[i]
#
#     return temp
#
#
# def calculate_euclidean(a1, a2):
#     return np.sqrt(np.sum((a1 - a2) ** 2))

from Research.DataAnalysis import LoadData

if __name__ == '__main__':
    train_data = LoadData.Load_TrainData()
    train_data_labels = LoadData.Load_TrainDataLabels()
    digits_list = LoadData.get_seperate_digits(train_data,train_data_labels)
    centroid_list = LoadData.find_digits_average(digits_list)
    eucledian_datalist = LoadData.find_euclediandistance(centroid_list,digits_list)
    LoadData.get_mostdifferent_images(eucledian_datalist)
    print('test')