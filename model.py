import tensorflow as tf
from sklearn.utils import shuffle


class Model:
    def __init__(self, chars, num_hidden):
        self.chars = chars
        num_chars = len(self.chars)
        self.data = tf.placeholder(tf.float32, [None, None, num_chars])
        self.target = tf.placeholder(tf.float32, [None, num_chars])

        self.lstm = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        # dropout different for each cell
        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm, output_keep_prob=0.5)
        # initial_state = self.cell.zero_state(batch_size,  dtype=tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cell, self.data, dtype=tf.float32)
        self.last_output = self.final_state.h
        self.weight = tf.Variable(tf.truncated_normal(
            [num_hidden, int(self.target.get_shape()[1])]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))
        self.prediction = tf.nn.softmax(tf.matmul(self.last_output, self.weight) + self.bias)
        self.cross_entropy = - \
                tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.prediction, 1e-10, 1.0)))
        self.optimizer = tf.train.AdamOptimizer()
        self.minimize = self.optimizer.minimize(self.cross_entropy)
        self.mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session( config=config)

        self.saver = tf.train.Saver()
    
    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def train(self, features, labels, features_val, labels_val, num_epochs, batch_size):


        losses = []
        val_loss = []
        for i in range(num_epochs):
            shuffled_features, shuffled_labels = shuffle(features, labels)
            self.run_epoch(batch_size, shuffled_features, shuffled_labels)
            incorrect = self.sess.run(self.error, {self.data: shuffled_features, self.target: shuffled_labels})
            incorrect_val = self.sess.run(self.error, {self.data: features_val, self.target: labels_val})
            print('Epoch {:2d} TrainError {:3.1f}%'.format(i, 100 * incorrect))
            print('ValError {:3.1f}%'.format(100 * incorrect_val))
            losses.append(incorrect)
            val_loss.append(incorrect_val)
            self.saver.save(self.sess, 'static/model/sess.ckpt')
        return losses, val_loss
            
    def run_epoch(self, batch_size, shuffled_features, shuffled_labels):
        num_of_batches = int(len(shuffled_features) / batch_size)
        idx = 0
        for i in range(num_of_batches):
            # Construct batch
            inp = shuffled_features[idx: idx + batch_size]
            out = shuffled_labels[idx: idx + batch_size]
            idx += batch_size

            self.sess.run(self.minimize, {self.data: inp, self.target: out})

    def load(self, path):
        self.saver.restore(self.sess, path)

    def predict(self, sequence):
        return self.sess.run(self.prediction, {self.data: sequence})
            