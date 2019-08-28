from flask import Flask, render_template, request, redirect, url_for, jsonify
from .predict import predict_next_word
from .model import Model
import numpy as np


app = Flask(__name__)

num_hidden = 128
chars = np.load("static/model/chars.npy")
our_model = Model(chars=chars, num_hidden=num_hidden)
our_model.load('static/model/sess.ckpt')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/user_input', methods=['GET', 'POST'])
def handle_input():
    message = request.form['input']
    app.logger.info(message)
    parsed = predict_next_word(message, model=our_model,
                     chars=chars, num_hidden=num_hidden)
    app.logger.info(type(parsed))
    return jsonify(username=parsed)


app.run(debug=True)
