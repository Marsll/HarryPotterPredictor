import numpy as np


def one_hot_features(features, chars):
    num_chars = len(chars)
    feature_num = []
    for strings in features:
        string_num = []
        for char in strings:
            temp_list = [0]*num_chars
            index = np.where(np.array(chars) == char)[0][0]
            temp_list[index] = 1
            string_num.append(temp_list)
        feature_num.append(string_num)
    return feature_num

# one hot labels
def one_hot_labels(labels, chars):
    num_chars = len(chars)
    labels_num = []
    for letter in labels:
        temp_list = [0]*num_chars
        index = np.where(np.array(chars) == letter)[0][0]
        temp_list[index] = 1
        labels_num.append(temp_list)
    return labels_num
   