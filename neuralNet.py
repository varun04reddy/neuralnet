import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# creating a neural network model with no ML libraries just math training and testing the MNIST data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y_train = train_data.iloc[:, 0].values
x_train = train_data.iloc[:, 1:].values / 255.0 

y_test = test_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values / 255.0 

x_train = x_train.T
x_test = x_test.T

def initialize_params():
    w1 = np.random.randn(10, 784) * np.sqrt(1. / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(1. / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

def softmax(z):
    z -= np.max(z, axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

# Forward propagation
def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((y.max() + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    m = x.shape[1] 

    one_hot_y = one_hot(y)
    dZ2 = a2 - one_hot_y
    dW2 = 1 / m * dZ2.dot(a1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = w2.T.dot(dZ2) * relu_deriv(z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = initialize_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dW1, db1, dW2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(a2)
            print("Iteration:", i, "Accuracy:", get_accuracy(predictions, y))
    return w1, b1, w2, b2

#W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.1, 500)

#_, _, _, a2_test = forward_prop(W1, b1, W2, b2, x_test)
#predictions_test = get_predictions(a2_test)
#print("Test Accuracy:", get_accuracy(predictions_test, y_test))

print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)


def display_image_from_csv(csv_file, image_index):
    data = pd.read_csv(csv_file)
    image = data.iloc[image_index, 1:].values
    label = data.iloc[image_index, 0]
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

image_index = 20  
display_image_from_csv('train.csv' , image_index)

