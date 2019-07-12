import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import flask
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader

app = flask.Flask(__name__)

pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def result():
    if flask.request.method == 'POST':
        result = flask.request.form

        data_dict = {
            'positive_affect': result['positive_affect'],
            'negative_affect': result['negative_affect'],
            'social_support': result['social_support'],
            'freedom': result['freedom'],
            'corruption': result['corruption'],
            'generosity': result['generosity'],
            'log_of_gdp_per_capita': result['log_of_gdp_per_capita'],
            'healthy_life_expectancy': result['healthy_life_expectancy']
            }

        data = pd.DataFrame(data_dict, index=[0])

        pred = pipe.predict(data)[0]
        prediction = int(round(pred))
        return render_template('result.html', prediction=prediction, data=data_dict)

if __name__ == '__main__':

    HOST = '127.0.0.1'
    PORT = 4001

    app.run(HOST, PORT)

    #  Debugger PIN: 315-047-346
