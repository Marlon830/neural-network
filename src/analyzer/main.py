#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## B-CNA-500-MPL-5-1-neuralnetwork-marlon.pegahi
## File description:
## main.py
##

import sys
import numpy as np
from src.analyzer.error_gestion import error_gestion
from src.NeuralNetwork.MyNeuralNetwork import MyNeuralNetwork

def parse_fen_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        X = []
        y = []
        result_mapping = {
            "Nothing": 0,
            "Stalemate": 1,
            "Check White": 2,
            "Check Black": 3,
            "Checkmate White": 4,
            "Checkmate Black": 5
        }
        for line in lines:
            parts = line.strip().split(' ')
            fen = parts[:6]
            board_matrix = fen_to_matrix(fen)
            X.append(board_matrix)
            try:
                result = parts[6]
                if len(parts) > 7:
                    result += " " + parts[7]
                y_encoded = [0, 0, 0, 0, 0, 0]
                y_encoded[result_mapping[result]] = 1
                y.append(y_encoded)
            except:
                y.append([0, 0, 0, 0, 0, 0])
        # print(X)
        # print(y)
        X = np.array(X)
        y = np.array(y)
        X = X.T / 1111.0
        y = y.T
    return X, y

def fen_to_matrix(fen):
    piece_placement, turn, castling, en_passant, halfmove, fullmove = fen
    rows = piece_placement.split('/')
    matrix = []
    for row in rows:
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    matrix.append(0)
            else:
                matrix.append(ord(char))

    matrix.append(ord(turn))

    if castling == '-':
        matrix.append(0)
    else:
        int_castling = 0
        castle_list = ['K', 'Q', 'k', 'q']
        for i in range(len(castle_list)):
            if castle_list[i] in castling:
                int_castling += 1 * 10**i
        matrix.append(int_castling)
    
    if en_passant == '-':
        matrix.append(0)
    else:
        matrix.append(ord(en_passant[0]) * 10 + ord(en_passant[1]))
    
    matrix.append(int(halfmove))
    matrix.append(int(fullmove))
    
    return np.array(matrix)

def train_neural_network(load_file, file, save_file):
    X, y = parse_fen_file(file)

    neural_network = MyNeuralNetwork(X, y, 0.1, 100)
    neural_network.load_perceptron(load_file)

    i = 0
    while True:
        if neural_network.accuracy_checking():
            print("Accuracy checking passed at epoch", i * neural_network.epoch)
            break
        neural_network.train()
        i += 1

        if neural_network.accuracy[-1] >= 0.99:
            neural_network.learning_rate = 0.5

        # if i % 20 == 0:
            # neural_network.save_perceptron(save_file + f"_{i * neural_network.epoch}")
        neural_network.save_perceptron(save_file)
        print(f"Epoch {i * neural_network.epoch}: Loss = {neural_network.loss[-1]}, Accuracy = {neural_network.accuracy[-1]}")

    neural_network.save_perceptron(save_file)

    # if "-p" in sys.argv or "--plot" in sys.argv:
    #     plt.figure()
    #     plt.plot(neural_network.loss)
    #     plt.title('Loss over epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')

    #     plt.figure()
    #     plt.plot(neural_network.accuracy, color='orange')
    #     plt.title('Accuracy over epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.show()

def predict_neural_network(load_file, file):
    X, y = parse_fen_file(file)
    result_mapping = {
        0: "Nothing",
        1: "Stalemate",
        2: "Check White",
        3: "Check Black",
        4: "Checkmate White",
        5: "Checkmate Black"
    }

    neural_network = MyNeuralNetwork(X, y, 1, 100)
    neural_network.load_perceptron(load_file)

    prediction = neural_network.predict()
    # neural_network.accuracy_score()
    # print(neural_network.accuracy[-1])
    for i, pred in enumerate(prediction):
        # print(f"{i + 1}: {result_mapping[pred]}")
        print(result_mapping[pred])

def main():
    error_gestion()
    argv = sys.argv
    mode = argv[1]
    load_file = ""
    file = ""
    save_file = ""
    if mode == "--predict" or (mode == "--train" and len(argv) == 4):
        load_file = argv[2]
        file = argv[3]
        save_file = load_file
    elif mode == "--train":
        load_file = argv[4]
        file = argv[5]
        save_file = argv[3]
    if mode == "--train":
        # print(f"Training neural network from {load_file} on {file} and saving it to {save_file}")
        train_neural_network(load_file, file, save_file)
    else:
        # print(f"Predicting with neural network from {load_file} on {file}")
        predict_neural_network(load_file, file)

if __name__ == '__main__':
    main()
