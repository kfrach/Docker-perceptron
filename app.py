from flask import Flask, jsonify, request
import numpy as np
import random

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1 + X.shape[1])]
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] = [w + update * x for w, x in zip(self.w_[1:], xi)]
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Create a flask app
app = Flask(__name__)

# Create an API end point
@app.route('/')
def say_hello():
    return "Hello World"

@app.route("/api/v1.0/predict", methods=['GET'])
def fun():
    x1 = request.args.get("x1", 0, type=float)
    x2 = request.args.get("x2", 0, type=float)
    features = [x1, x2]
    
    perceptron = Perceptron()
    perceptron.w_ = [0.5, 0.5, 0.5]
    
    predicted_class = perceptron.predict(np.array(features))
    return jsonify(features={'x1': x1, 'x2': x2}, predicted_class=int(predicted_class))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
