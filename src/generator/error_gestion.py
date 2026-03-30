#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## B-CNA-500-MPL-5-1-neuralnetwork-marlon.pegahi
## File description:
## error_gestion.py
##

import sys

def error_gestion():
    argv = sys.argv
    if (len(argv) >= 2 and argv[1] == "--help"):
        print("USAGE")
        print("    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n")
        print("DESCRIPTION")
        print("    config_file_i\t Configuration file containing description of a neural network we want to generate.")
        print("    nb_i\t\t Number of neural networks to generate based on the configuration file.")
        sys.exit(0)
    if len(argv) == 1 or len(argv) % 2 == 0:
        print("Invalid number of arguments", file=sys.stderr)
        sys.exit(84)
    try:
        for i in range(2, len(argv), 2):
            value = int(argv[i])
            if value < 1:
                raise ValueError()
    except ValueError as e:
        print("Number of neural networks must be an integer greater than or equal to 1", file=sys.stderr)
        sys.exit(84)
    except:
        print("Invalid argument", file=sys.stderr)
        sys.exit(84)
