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
            y_value = int(line[-1])
            y_encoded = [0, 0, 0]
            y_encoded[y_value] = 1
            y.append(y_encoded)
        # print(X)
        # print(y)
        # sys.exit(0)
        X = np.array(X)
        y = np.array(y)
    X = X.T
    y = y.T
    # y = y.reshape((1, y.shape[0]))


    new_perceptron = MyNeuralNetwork.MyNeuralNetwork([18, 18], X, y, 1, 100)
    i = 0
    while True:
        if new_perceptron.accuracy_checking():
            print("Accuracy checking passed at epoch", i)
            break
        new_perceptron.train()
        i += 1

        if new_perceptron.accuracy[-1] >= 0.99:
            new_perceptron.learning_rate = 0.5

        if i % 20 == 0:
            new_perceptron.save_perceptron(file)
            print(f"Epoch {i * new_perceptron.epoch}: Loss = {new_perceptron.loss[-1]}, Accuracy = {new_perceptron.accuracy[-1]}")

    new_perceptron.save_perceptron(file)

    if "-p" in sys.argv or "--plot" in sys.argv:
        plt.figure()
        plt.plot(new_perceptron.loss)
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.figure()
        plt.plot(new_perceptron.accuracy, color='orange')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

if __name__ == '__main__':
    main()
