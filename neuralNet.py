import numpy as np
import pandas as pd
import matplotlib as plt



data = pd.read_csv('train.csv')

print(data.head)


data = np.array(data)

np.random.shuffle(data)

m,n = data.shape

dataDev = data[0:1000].T
yDev = dataDev[0]
xDev = dataDev[1:n]

trainData = data[1000:m].T
yTrain = trainData[0]
xTrain = trainData[1:n]

print(xTrain[:, 0].shape)

def params():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10,1)
    w2 = np.random.randn(10,10)
    b2 = np.random.randn(10,1)

    return w1, b1, w2, b2

def reLu(z):
    return np.maximum(0, z)

def softMax(z):
    z -= np.max(z, axis=0)  # Subtracting the max for numerical stability
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=0)


def forwardProp(w1, b1, w2, b2, x):
    Z1 = w1.dot(x) + b1
    A1 = reLu(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softMax(A1)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def ReLU_deriv(Z):
    return Z > 0

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(xTrain, yTrain, 0.10, 500)

