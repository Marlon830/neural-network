#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## B-CNA-500-MPL-5-1-neuralnetwork-marlon.pegahi
## File description:
## main.py
##

import numpy as np
import json
import sys
from src.generator.error_gestion import error_gestion

def generate_neural_network(config_file, nb):
    if not config_file.endswith('.conf'):
        raise ValueError("Config file must end with .conf")
    config_name = config_file[:-5]

    with open(config_file, 'r') as f:
        config = f.read().splitlines()
    
    config_dict = {}
    for line in config:
        key, value = line.split(' = ')
        config_dict[key] = eval(value)
    
    input_neurons = config_dict['input_neurons']
    hidden_layers = config_dict['hidden_layers']
    output_neurons = config_dict['output_neurons']

    layer_dimensions = [input_neurons] + hidden_layers + [output_neurons]
    for i in range(nb):
        weight = []
        bias = []
        gamma = []
        beta = []
        for j in range(1, len(layer_dimensions)):
            weight.append(np.random.randn(layer_dimensions[j], layer_dimensions[j - 1]) * np.sqrt(2. / layer_dimensions[j - 1]))
            bias.append(np.zeros((layer_dimensions[j], 1)))
            if j < len(layer_dimensions) - 1:
                gamma.append(np.ones((layer_dimensions[j], 1)))
                beta.append(np.zeros((layer_dimensions[j], 1)))
        with open(f"{config_name}_{i + 1}.json", "w") as f:
            state = {
                'W': [w.tolist() for w in weight],
                'b': [b.tolist() for b in bias],
                'gamma': [g.tolist() for g in gamma],
                'beta': [b.tolist() for b in beta]
            }
            json.dump(state, f, indent=4)

def main():
    error_gestion()

    argv = sys.argv
    for i in range(1, len(argv), 2):
        config_file = argv[i]
        nb = int(argv[i + 1])
        try:
            generate_neural_network(config_file, nb)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(84)
    pass

if __name__ == '__main__':
    main()
