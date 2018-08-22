from model import Model
import numpy as np

# Load data
path = "./static/data/hp1.txt"
with open(path) as f:  # Use file to refer to the file object

    data = f.read()[:40000]
    print(len(data))
    chars = sorted(list(set(data)))
    print(chars)
    num_chars = len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    sequence_length = 40
    step = 30
    text_slices = []
    text_slice_labels = []
    for i in range(0, len(data) - sequence_length, step):
        text_slices.append(data[i: i + sequence_length])
        text_slice_labels.append(data[i + sequence_length])
    print(f'num training examples: {len(text_slices)}')
    features = np.array(text_slices)
    print(features)
    labels = np.array(text_slice_labels)

    def one_hot_features(features):
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
    def one_hot_labels(labels):
        labels_num = []
        for letter in labels:
            temp_list = [0]*num_chars
            index = np.where(np.array(chars) == letter)[0][0]
            temp_list[index] = 1
            labels_num.append(temp_list)
        return labels_num




x = np.array(one_hot_features(features))
y = np.array(one_hot_labels(labels))

split = int(len(x)*0.8)




# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

batch_size = 11
num_hidden = 40
num_epochs = 10
features = x[:split]
labels = y[:split]

features_val = x[split:]
labels_val = y[split:]
test_in = x


our_model = Model(num_chars=num_chars, num_hidden=num_hidden, batch_size=batch_size)
our_model.train(features, labels, features_val, labels_val, num_epochs, batch_size)