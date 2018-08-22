from flask import Flask, render_template, request, redirect, url_for, jsonify


app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/user_input', methods=['GET', 'POST'])
def handle_input():
    message = request.form['input']
    return jsonify(username=message)


app.run(debug=True)