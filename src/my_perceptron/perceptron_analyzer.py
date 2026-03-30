#!/usr/bin/env python3

import NeuralNetwork.MyNeuralNetwork as MyNeuralNetwork
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    file = sys.argv[1]
    dataset = sys.argv[2]
    with open(dataset, 'r') as f:
        lines = f.readlines()
        X = []
        y = []
        for line in lines:
            line = line.strip().split(',')
            l = []
            for i in range(len(line[0])):
                l.append(int(line[0][i]))
            X.append(l)
            y.append([int(line[-1])])
        # print(X)
        # print(y)
        X = np.array(X)
        y = np.array(y)
    # sys.exit(0)
    # X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    # y = np.array([[0], [1], [1], [0]])
    X = X.T
    y = y.reshape((1, y.shape[0]))

    # file = sys.argv[1]
    # X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    # X = X.T

    perceptron = MyNeuralNetwork.MyNeuralNetwork([18, 18], X)
    perceptron.load_perceptron(file)
    print(perceptron.predict())


if __name__ == '__main__':
    main()
