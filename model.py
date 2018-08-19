import tensorflow as tf
from tensorflow.contrib.feature_column import sequence_categorical_column_with_hash_bucket
from tensorflow.feature_column import embedding_column
from tensorflow.contrib.estimator import RNNEstimator
import numpy as np

#data

path = "./static/data/hp1.txt"
with open(path) as f: # Use file to refer to the file object

    data = f.read()
    print(len(data))
    chars = sorted(list(set(data)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    sequence_length = 40
    step = 3
    text_slices = []
    text_slice_labels = []
    for i in range(0, len(data) - sequence_length, step):
        text_slices.append(data[i: i + sequence_length])
        text_slice_labels.append(data[i + sequence_length])
    print(f'num training examples: {len(text_slices)}')
    features = np.array(text_slices).astype(np.str)
    labels = np.array(text_slice_labels).astype(np.str)
# should be calc not hard coded
number_of_categories = len(chars)
batch_size = 32

token_sequence = sequence_categorical_column_with_hash_bucket(
    key="text", hash_bucket_size=number_of_categories, dtype=tf.string)
# what does this even mean???
token_emb = embedding_column(categorical_column=token_sequence, dimension=80)
    

def rnn_cell_fn(mode):
    cells = [tf.contrib.rnn.LSTMCell(size) for size in [32, 16]]
    if mode == tf.estimator.ModeKeys.TRAIN:
        cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
                 for cell in cells]
    return tf.contrib.rnn.MultiRNNCell(cells)


estimator = RNNEstimator(
    head=tf.contrib.estimator.regression_head(),
    sequence_feature_columns=[token_emb],
    rnn_cell_fn=rnn_cell_fn)


# Input builders


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"text": features},
    y=labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)

estimator.train(input_fn=train_input_fn, steps=100)


def input_fn_eval():  # returns x, y
    pass


metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)


def input_fn_predict():  # returns x, None
    pass


predictions = estimator.predict(input_fn=input_fn_predict)
