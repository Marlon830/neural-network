import numpy as np
import json

class MyNeuralNetwork:
    def __init__(self, X, y=np.array([]), learning_rate=0.01, epoch=100, batch_size=128):
        self.W = []
        self.b = []
        self.X = X
        self.X_temp = X
        self.y = y
        self.y_temp = y
        self.loss = []
        self.accuracy = []
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.nb_of_layer = 0
        self.A = []
    
    def relu_activation_function(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid_activation_function(self, Z):
        return 1 / (1 + np.exp(-Z))

    # def softmax_activation_function(self, Z):
    #     return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def softmax_activation_function(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self):
        self.A[0] = self.X

        for i in range(1, self.nb_of_layer):
            Z = self.W[i - 1].dot(self.A[i - 1]) + self.b[i - 1]
            if i == self.nb_of_layer - 1:
                self.A[i] = self.softmax_activation_function(Z)
            else:
                self.A[i] = self.relu_activation_function(Z)
    
    # for multi class classification
    def cross_entropy(self, A):
        self.loss.append(-np.sum(self.y * np.log(A[-1] + 1e-8)))

    # for binary classification
    def log_loss(self):
        self.loss.append(-(1 / len(self.y)) * np.sum(self.y * np.log(self.A[-1]) - (1 - self.y) * np.log(1 - self.A[-1])))

    def back_propagation(self):
        dZ = self.A[-1] - self.y
        dW = []
        db = []

        for i in reversed(range(1, self.nb_of_layer)):
            dW.insert(0, 1 / self.y.shape[1] * np.dot(dZ, self.A[i - 1].T))
            db.insert(0, 1 / self.y.shape[1] * np.sum(dZ, axis=1, keepdims=True))
            # dZ = np.dot(self.W[i - 1].T, dZ) * self.A[i - 1] * (1 - self.A[i - 1])
            if i > 1:
                dZ = np.dot(self.W[i - 1].T, dZ) * (self.A[i - 1] > 0)
        
        for i in range(1, self.nb_of_layer):
            self.W[i - 1] -= self.learning_rate * dW[i - 1]
            self.b[i - 1] -= self.learning_rate * db[i - 1]

    def predict(self):
        self.forward_propagation()
        return np.argmax(self.A[-1], axis=0)

    def accuracy_score(self):
        y_pred = self.predict()
        y_real = np.argmax(self.y, axis=0)
        self.accuracy.append(np.sum(y_pred == y_real) / len(y_real))

    # def train(self):
    #     for i in tqdm(range(self.epoch)):
    #         self.forward_propagation()
    #         self.back_propagation()
    #     self.cross_entropy()
    #     self.accuracy_score()

    def train(self):
        accumulated_A = []
        for _ in range(self.epoch):
            self.X_temp = self.X
            self.y_temp = self.y
            accumulated_A = []
            for X_batch, y_batch in self.create_batches(self.X, self.y, self.batch_size):
                self.X = X_batch
                self.y = y_batch
                self.forward_propagation()
                self.back_propagation()
                accumulated_A.append(self.A[-1])
            self.X = self.X_temp
            self.y = self.y_temp
            accumulated_A = np.concatenate(accumulated_A, axis=1)
        self.cross_entropy(accumulated_A)
        self.accuracy_score()
    
    def create_batches(self, X, y, batch_size):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]
        for k in range(0, m, batch_size):
            X_batch = X_shuffled[:, k:k + batch_size]
            y_batch = y_shuffled[:, k:k + batch_size]
            yield X_batch, y_batch

    def accuracy_checking(self):
        y_pred = self.predict()
        y_real = np.argmax(self.y, axis=0)
        return np.array_equal(y_pred, y_real)

    def save_perceptron(self, filename):
        with open(filename, 'w') as f:
            state = {
                'W': [w.tolist() for w in self.W],
                'b': [b.tolist() for b in self.b]
            }
            json.dump(state, f, indent=4)
    
    def load_perceptron(self, filename):
        with open(filename, 'r') as f:
            state = json.load(f)
            self.W = [np.array(w) for w in state['W']]
            self.b = [np.array(b) for b in state['b']]
            self.nb_of_layer = len(self.W) + 1
            self.A = [0 for _ in range(self.nb_of_layer)]
