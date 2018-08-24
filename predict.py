import tensorflow as tf
import numpy as np
from model import Model
from data_processing import one_hot_features, one_hot_labels


num_hidden = 128
chars = np.load("static/model/chars.npy")

def predict_next_word(sequence, chars, num_hidden):
    our_model = Model(chars=chars, num_hidden=num_hidden)
    our_model.load('static/model/sess.ckpt')
    sequence = np.array(sequence)[ np.newaxis]
    one_hot_sequence = one_hot_features(sequence, chars)
    counter = 0
    next_char = predict_next_char(our_model, one_hot_sequence)
    while chars[np.argmax(next_char)] != " " and counter < 10:
        counter += 1
        next_char = predict_next_char(our_model, one_hot_sequence)
        one_hot_sequence = np.append(np.array(one_hot_sequence), next_char[:, np.newaxis, :], axis=1)
        print(chars[np.argmax(next_char)], next_char[:, np.argmax(next_char)])
    print(chars[np.argmax(one_hot_sequence, axis=2)])

def predict_next_char(model, one_hot_sequence):

    next_char = model.predict(one_hot_sequence)
    return next_char


predict_next_word("Harry P", chars, num_hidden)