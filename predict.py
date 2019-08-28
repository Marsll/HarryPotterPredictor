import tensorflow as tf
import numpy as np
from .model import Model
from .data_processing import one_hot_features, one_hot_labels


def predict_next_word(sequence, model, chars, num_hidden):
    sequence_arr = np.array(sequence)[np.newaxis]
    one_hot_sequence = one_hot_features(sequence_arr, chars)
    counter = 0
    next_char = predict_next_char(model, one_hot_sequence)
    predictions = []
    probs = []
    while chars[np.argmax(next_char)] != " " and counter < 10:
        counter += 1
        next_char = predict_next_char(model, one_hot_sequence)
        one_hot_sequence = np.append(
            np.array(one_hot_sequence), next_char[:, np.newaxis, :], axis=1)
        predictions.append(chars[np.argmax(next_char)])
        probs.append(next_char[:, np.argmax(next_char)][0])
    for c, num in zip(predictions, probs):
        print(c+":", num)

    return sequence + "".join(predictions)


def predict_next_char(model, one_hot_sequence):
    next_char = model.predict(one_hot_sequence)
    return next_char


# num_hidden = 128
# chars = np.load("static/model/chars.npy")
# our_model = Model(chars=chars, num_hidden=num_hidden)
# our_model.load('static/model/sess.ckpt')
# print(predict_next_word("Harry P", our_model, chars, num_hidden))
