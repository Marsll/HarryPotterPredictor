import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
import os
import shutil


out = '/tmp/rnn'
if os.path.exists(out):
    shutil.rmtree(out, ignore_errors=True)


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
num_hidden = 50
epoch = 10
train_input = x[:split]
train_output = y[:split]
# change!
test_input = x[split:]
test_output = y[split:]
test_in = x

# batchsize, number of inputs, dimension of each input
data = tf.placeholder(tf.float32, [None, None, char_len])
target = tf.placeholder(tf.float32, [None, char_len])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
initial_state = cell.zero_state(batch_size,  dtype=tf.float32) #initialize with zeros
val, state = tf.nn.dynamic_rnn(cell, data, initial_state=initial_state, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
# last = tf.gather(val, int(val.get_shape()[0]) - 1) #get last output
# last = tf.gather(val,  [-1]) #get last output
#last = val.__getitem__(-1)  # get last output
last = state.h # get last output
weight = tf.Variable(tf.truncated_normal(
    [num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = - \
    tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

#init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
saver = tf.train.Saver()

no_of_batches = int(len(features) / batch_size)
losses = []
val_loss = []

counter = 0

# should shuffle data
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr +
                               batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize, {data: inp, target: out})
        #save the model
        saver.save(sess, 'static/model/sess.ckpt')
    incorrect = sess.run(error, {data: train_input, target: train_output})
    incorrect_val = sess.run(error, {data: test_input, target: test_output})
    print('Epoch {:2d} TrainError {:3.1f}%'.format(i, 100 * incorrect))
    print('ValError {:3.1f}%'.format(100 * incorrect_val))
    losses.append(incorrect)
    val_loss.append(incorrect_val)


#incorrect = sess.run(error, {data: test_input, target: test_output})
#print('Epoch {:2d} ValidationError {:3.1f}%'.format(i + 1, 100 * incorrect))
pred = sess.run(prediction, {data: train_input})
pred2 = sess.run(prediction, {data: test_input})
labels = sess.run(prediction, {data: test_in})
sess.close()

# print training and validation error
plt.plot(np.arange(len(losses)) + 1, losses, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)

plt.figure()
plt.plot(np.arange(len(val_loss)) + 1, val_loss, 'ro', label='Validation loss')
plt.title('Accuracy')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
plt.figure()

#plt.show()

prediction = np.argmax(labels, axis=1)

#db.set_trace()
