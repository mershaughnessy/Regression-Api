import json
import numpy as np
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/linearRegression', methods=['POST'])
def perform_linear_regression():
    dataset = json.loads(request.data)
    x_train, x_test, y_train, y_test = train_test_split(dataset['IndependentVariable'],dataset['DependentVariable'] , test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(np.array(x_train).reshape(-1, 1), np.array(y_train))
    y_pred = regressor.predict(np.array(x_test).reshape(-1,1))

    return jsonify({'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test': y_test, 'y_predition': y_pred.tolist()})

if __name__ == '__main__':
   app.run(port=5000)