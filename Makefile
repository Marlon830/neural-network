##
## EPITECH PROJECT, 2024
## B-CNA-500-MPL-5-1-neuralnetwork-marlon.pegahi
## File description:
## Makefile
##

SRC_GENERATOR =  src/generator/main.py
SRC_ANALYZER = src/analyzer/main.py

TARGET_GENERATOR = my_torch_generator
TARGET_ANALYZER = my_torch_analyzer

all: $(TARGET_GENERATOR) $(TARGET_ANALYZER)

$(TARGET_GENERATOR): $(SRC_GENERATOR)
		cp $(SRC_GENERATOR) $(TARGET_GENERATOR)
		chmod +x $(TARGET_GENERATOR)

$(TARGET_ANALYZER): $(SRC_ANALYZER)
		cp $(SRC_ANALYZER) $(TARGET_ANALYZER)
		chmod +x $(TARGET_ANALYZER)

clean: fclean

fclean:
	$(RM) $(TARGET_GENERATOR)
	$(RM) $(TARGET_ANALYZER)

re : fclean all
