# MY_TORCH

> Made by **Marlon PEGAHI** and **Alexandre BRET**

## Project Overview
MY_TORCH is a project that involves creating and training a neural network to analyze chessboard states. The project consists of two main components:
1. **Neural Network Generator**: Generates neural networks based on configuration files.
2. **Chessboard Analyzer**: Trains and evaluates neural networks to predict the state of a chess game.

## Binaries
The project provides two binaries:
- `my_torch_generator`: Generates neural networks from configuration files.
- `my_torch_analyzer`: Analyzes chessboards using a neural network, either in training or prediction mode.

## Compilation
The project includes a Makefile to handle compilation. The Makefile supports the following rules:
- `make`: Compiles the project.
- `make re`: Recompiles the project.
- `make clean`: Removes object files.
- `make fclean`: Removes object files and binaries.

## Usage

### Neural Network Generator
To generate neural networks, use the `my_torch_generator` binary with the following syntax:
```sh
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```
- `config_file_i`: Path to the configuration file for the neural network.
- `nb_i`: Number of neural networks to generate based on the configuration file.

Exemple:
```sh
./my_torch_generator basic_network.conf 3
```

### Chessboard Analyzer
To analyze chessboards, use the `my_torch_analyzer` binary with the following syntax:
```sh
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE
```
- `--train`: Launches the neural network in training mode.
- `--predict`: Launches the neural network in prediction mode.
- `--save` SAVEFILE: Saves the trained neural network to `SAVEFILE` (only in training mode).
- `LOADFILE`: File containing the neural network.
- `FILE`: File containing chessboards in FEN notation.

Exemple for training:
```sh
./my_torch_analyzer --train --save my_trained_network my_torch_network_basic.json chessboards.txt
```

Exemple for prediction:
```sh
./my_torch_analyzer --predict my_torch_network_basic.json chessboards.txt
```

## Configuration Files
The configuration file for the neural network should contain the following parameters:
- `input_neurons`: Number of input neurons.
- `hidden_layers`: List of hidden layer sizes.
- `output_neurons`: Number of output neurons.

Exemple (`basic_network.conf`):
```conf
input_neurons = 519
hidden_layers = [128, 256]
output_neurons = 6
```

## Neural Network Functionality

### Initialization
The neural network is initialized with random weights and biases using the Kaiming initialization method. This method helps the network converge faster by preventing the vanishing gradient problem.

### Forward Propagation
Forward propagation is the process by which input data is passed through the neural network to generate an output. Each layer of the network applies a set of weights and biases to the input data, followed by an activation function. The output of one layer becomes the input to the next layer.

### Backward Propagation
Backward propagation, or backpropagation, is the process of adjusting the weights and biases of the network based on the error of the output. This is done by calculating the gradient of the loss function with respect to each weight and bias, and then updating them in the opposite direction of the gradient to minimize the loss.

### Loss Function
The loss function measures the difference between the predicted output of the network and the actual output. The Cross-Entropy loss function is commonly used in classification problems and is defined as: `loss = -sum(y_i * log(p_i) for i in range(n))`, where `y_i` is the actual output and `p_i` is the predicted output.

### Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Two common activation functions used in neural networks are ReLU and Softmax.

#### ReLU (Rectified Linear Unit)
ReLU is an activation function defined as `f(x) = max(0, x)`. It is simple and effective, helping the network to converge faster by mitigating the vanishing gradient problem.

#### Softmax
Softmax is an activation function often used in the output layer of a classification network. It converts the raw output scores into probabilities by applying the exponential function to each score and normalizing them so that they sum to 1. The formula for Softmax is: `softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in range(n))`.

### Training Process
1. **Initialization**: The network is initialized with random weights and biases using Kaiming initialization.
2. **Forward Propagation**: Input data is passed through the network to generate an output.
3. **Loss Calculation**: The difference between the predicted output and the actual output is calculated using the Cross-Entropy loss function.
4. **Backward Propagation**: The gradients of the loss with respect to each weight and bias are calculated using backpropagation.
5. **Weight Update**: The weights and biases are updated using an optimization algorithm (e.g., Gradient Descent) to minimize the loss.
6. **Iteration**: Steps 2-5 are repeated for a number of epochs or until the loss converges to an acceptable level.

## Benchmark

For detailed benchmarking results and performance analysis, please refer to the [benchmark documentation](doc/benchmark.md).