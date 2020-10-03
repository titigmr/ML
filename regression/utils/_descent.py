import numpy as np

def _prediction(X, beta):
    return X.dot(beta)

def _loss(X, y, beta):
    n = len(y)
    return 1/(2*n) * np.sum((_prediction(X, beta) - y)**2)

def _gradient(X, y, beta):
    n = len(y)
    return 1/n * X.T.dot(_prediction(X, beta) - y)

def gradient_descent(X, y, beta, learning_rate, n_iterations):
    cost_history = []

    for i in range(0, n_iterations):
        beta = beta - learning_rate * _gradient(X, y, beta)
        cost_history.append(_loss(X, y, beta))
    return beta, cost_history