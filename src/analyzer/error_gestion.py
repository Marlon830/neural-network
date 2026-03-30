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
        print("    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE\n")
        print("DESCRIPTION")
        print("    --train\t Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.")
        print("    --predict\t Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.")
        print("    --save\t Save neural network into SAVEFILE. Only works in train mode.\n")
        print("    LOADFILE\t  File containing an artificial neural network")
        print("    FILE\t  File containing chessboards")
        sys.exit(0)
    if len(argv) < 4 or len(argv) > 6:
        print("Invalid number of arguments", file=sys.stderr)
        sys.exit(84)
    if argv[1] not in ["--train", "--predict"]:
        print("Invalid mode", file=sys.stderr)
        sys.exit(84)
    if argv[1] == "--train":
        if len(argv) == 5:
            print("Missing or too many argument", file=sys.stderr)
            sys.exit(84)
        if len(argv) == 6 and argv[2] != "--save":
            print("Invalid argument", file=sys.stderr)
            sys.exit(84)
    if argv[1] == "--predict" and len(argv) != 4:
        print("Too many argument", file=sys.stderr)
        sys.exit(84)
