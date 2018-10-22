#!/usr/bin/env python3

from typing import List, Iterable
from pathlib import Path
import itertools
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


def clean_images(ascii_images: List[List[str]]) -> Iterable[List[List[bool]]]:
    width = max(map(len, itertools.chain(ascii_images)))
    height = max(map(len, ascii_images))
    for image in ascii_images:
        nonblank_rows = [[not char.isspace() for char in row.ljust(width)] for row in image]
        blank_rows = [list(itertools.repeat(False, width)) for _ in range(height - len(image))]
        yield nonblank_rows + blank_rows


def image_to_text(image: List[List[bool]]) -> str:
    return "\n".join("".join(
        map(lambda b: "X" if b else " ", row)) for row in image)


def train(training_files: List[str]):
    images = clean_images([Path(file).read_text().split("\n") for file in training_files])


def read_model() -> List[List[float]]:
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
