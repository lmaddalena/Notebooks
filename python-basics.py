import numpy as np
import math

def main():

    x = 3
    print("x = ", x)
    print("sigmoid(x): ", sigmoid(x))
    print()

    x = np.array([1,2,3])
    print("x = ", x)
    print("sigmoid(x): ", sigmoid(x))
    print()

    print("x = ", x)
    print("dsigmoid(x): ", dsigmoid(x))
    print

    x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
    print("x = ", x)
    print("normalize(x): ", normalize(x))
    print

    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0 ,0]])
    print("x = ", x)
    print("softmax(x): ", softmax(x))
    print

    return

def softmax(x):
    t = np.exp(x)
    s = np.sum(t, axis=1, keepdims=True)
    t = t/s
    return t

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def normalize(x):
    norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x/norm
    return x

if __name__ == '__main__':
    main()