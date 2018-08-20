import tensorflow as tf
import numpy as np

path = "./static/data/hp1.txt"
with open(path) as f:  # Use file to refer to the file object

    data = f.read()[:40000]
    print(len(data))
    chars = sorted(list(set(data)))
    print(chars)
    char_len = len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    sequence_length = 40
    step = 300
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
                temp_list = [0]*char_len
                index = np.where(np.array(chars) == char)[0][0]
                temp_list[index] = 1
                string_num.append(temp_list)
            feature_num.append(string_num)
        return feature_num

    # one hot labels
    def one_hot_labels(labels):
        labels_num = []
        for letter in labels:
            temp_list = [0]*char_len
            index = np.where(np.array(chars) == letter)[0][0]
            temp_list[index] = 1
            labels_num.append(temp_list)
        return labels_num




x = np.array(one_hot_features(features))
y = np.array(one_hot_labels(labels))

split = int(len(x)*0.8)




# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

batch_size = 10
num_hidden = 20
epoch = 4
train_input = x[:split]
train_output = y[:split]
# change!
test_input = x[split:]
test_output = y[split:]
test_in = x


#model
batch_size = 10
num_hidden = 20
epoch = 4
#hard coded!!!!!!!!
char_len = 63

data = tf.placeholder(tf.float32, [None, None, char_len])
target = tf.placeholder(tf.float32, [None, char_len])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = val.__getitem__(-1)  # get last output
weight = tf.Variable(tf.truncated_normal(
    [num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)



#laod model
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, 'static/model/sess.ckpt')
labels = sess.run(prediction, {data: test_in})
print(labels)
sess.close()
