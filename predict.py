import tensorflow as tf
import numpy as np
from model import Model
from data_processing import one_hot_features, one_hot_labels


num_hidden = 128
chars = np.load("static/model/chars.npy")

def predict_next_word(sequence, chars):
    our_model = Model(chars=chars, num_hidden=num_hidden)
    our_model.load('static/model/sess.ckpt')
    sequence = np.array(sequence)[ np.newaxis]
    one_hot_sequence = one_hot_features(sequence, chars)
    next_char = predict_next_char(our_model, one_hot_sequence)
    print(next_char.shape)
    print(chars[np.argmax(next_char)], next_char[:,np.argmax(next_char)])

def predict_next_char(model, one_hot_sequence):

    next_char = model.predict(one_hot_sequence)
    return next_char


predict_next_word("Hagrid it was ", chars)