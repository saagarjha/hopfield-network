#!/usr/bin/env python3

from functools import reduce
from typing import List
from pathlib import Path
import sys


def invalid_usage():
	print("""Usage:

hopsfield-network.py train <training-files...>
	Train the Hopsfield network on the specified files, which are expected to be
	ASCII "images". The generated model is printed to standard output.

hopsfield-network.py match <image-file>
	Find the best match, as determined from the model taken from standard input,
	for the specified file (an ASCII "image"). The match is printed to standard
	output.

Note: Images are "black-and-white" in the sense that whitespace (other than
newlines, which indicate dimensions) is interpreted as "white" and any other
character is considered to be "black". Nonrectangular "ragged" images will be
padded with a "white" background. Output will use a space (' ') to represent
"white" and an 'X' to represent "black.""", file=sys.stderr)
	sys.exit(1)


def clean_images(ascii_images: List[str]) -> List[List[List[bool]]]:
	width = max(map(len, reduce(list.__add__, ascii_images)))  # where's my flatmap, sigh
	height = max(map(len, ascii_images))
	return [[[not character.isspace() for character in row.ljust(width)] for row in image] +
         [[False] * width for i in range(height - len(image))] for image in ascii_images]


def train(training_files: [str]):
	images = clean_images([Path(file).read_text().split("\n") for file in training_files])


def read_model() -> [[float]]:
	model_size = int(input())
	weights = [([0] * model_size) for i in range(model_size)]
	for neuron1 in range(0, model_size):
		for neuron2 in range(neuron1 + 1, model_size):
			weights[neuron1][neuron2] = weights[neuron2][neuron1] = float(input())
	return weights


def match(image_file: str):
	model = read_model()


if len(sys.argv) <= 1:
	invalid_usage()
if sys.argv[1] == "train":
	train(sys.argv[2:])
elif sys.argv[1] == "match":
	match(sys.argv[2])
else:
	invalid_usage()
