from .model import Model
import numpy as np
from .data_processing import one_hot_features, one_hot_labels
import os
# Load data
path = "./static/data/hp1.txt"
with open(path) as f:  # Use file to refer to the file object

    data = f.read()
    print(len(data))
    chars = sorted(list(set(data)))
    print(chars)
    data = data[100000:200000]
    num_chars = len(chars)
    #char_indices = dict((c, i) for i, c in enumerate(chars))
    #indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    sequence_length = 40
    step = 3
    text_slices = []
    text_slice_labels = []
    for i in range(0, len(data) - sequence_length, step):
        text_slices.append(data[i: i + sequence_length])
        text_slice_labels.append(data[i + sequence_length])
    print(f'num training examples: {len(text_slices)}')
    features = np.array(text_slices)
    
    labels = np.array(text_slice_labels)
    print(features[205:209], labels[205:209])


np.save("static/model/chars", np.array(chars))


x = np.array(one_hot_features(features, chars))
y = np.array(one_hot_labels(labels, chars))

print(x.shape)
split = int(len(x)*0.8)




# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

batch_size = 100
num_hidden = 128
num_cells = 3 #implement
num_epochs = 30
features = x[:split]
labels = y[:split]

features_val = x[split:]
labels_val = y[split:]
test_in = x


our_model = Model(chars=chars, num_hidden=num_hidden)
path = 'static/model/sess.ckpt.index'
if os.path.exists(path):
    our_model.load("static/model/sess.ckpt")
else:
    print("initalized")
    our_model.initialize()
our_model.train(features, labels, features_val, labels_val, num_epochs, batch_size)
