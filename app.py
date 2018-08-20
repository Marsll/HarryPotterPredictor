from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/user_input', methods=['GET', 'POST'])
def handle_input():
    email = request.form
    app.logger.info(email)
    return redirect(url_for('index'))


app.run(debug=True)