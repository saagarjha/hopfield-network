#!/usr/bin/env python3

from typing import List, Iterable, Tuple
from pathlib import Path
import itertools
import sys


def invalid_usage():
    print("""Usage:

hopsfield-network.py train <training-files...>
    Train the Hopsfield network on the specified files, which are expected to be
    ASCII "images". The generated model is printed to standard output.

hopsfield-network.py match <image-file> <training-files...>
    Find the best match, as determined from the model taken from standard input,
    for the specified file (an ASCII "image"). The match is printed to standard
    output.

Note: Images are "black-and-white" in the sense that whitespace (other than
newlines, which indicate dimensions) is interpreted as "white" and any other
character is considered to be "black". Nonrectangular "ragged" images will be
padded with a "white" background. Output will use a space (' ') to represent
"white" and an 'X' to represent "black.""", file=sys.stderr)
    sys.exit(1)

def clean_image(image: List[str], width: int, height: int) -> List[List[bool]]:
    # Strip trailing blank lines
    image = list(itertools.dropwhile(lambda s: not s, image[::-1]))[::-1]
    assert(len(image) <= height)
    nonblank_rows = [[not char.isspace() for char in row.ljust(width)] for row in image]
    blank_rows = [list(itertools.repeat(False, width)) for _ in range(height - len(image))]
    return nonblank_rows + blank_rows


def clean_images(ascii_images: List[List[str]]) -> Tuple[List[List[List[bool]]], int, int]:
    """Returns the cleaned images, along with the width and height of each image"""
    width = max(map(len, itertools.chain.from_iterable(ascii_images)))
    height = max(map(len, ascii_images))
    return [clean_image(image, width, height) for image in ascii_images], width, height


def image_to_text(image: List[List[bool]]) -> str:
    """Converts image data to a string"""
    return "\n".join("".join(
        map(lambda b: "X" if b else " ", row)) for row in image)


def image_to_data(image: List[List[bool]]) -> Iterable[int]:
    for row in image:
        for value in row:
            yield 1 if value else -1


def train(images: List[List[List[bool]]]) -> List[List[int]]:
    assert(images)
    image_height = len(images[0])
    image_width = len(images[0][0])
    neuron_count = image_width * image_height
    print("Image size is {} x {}".format(image_width, image_height))
    weights = [list(itertools.repeat(0, neuron_count))
               for _ in range(0, neuron_count)]
    img_nums = list(map(lambda img: list(image_to_data(img)), images))
    for i in range(0, neuron_count):
        for j in range(i + 1, neuron_count):
            weights[i][j] = weights[j][i] = (
                sum(img_data[i] * img_data[j] for img_data in img_nums))
    # print("Weights:")
    # for i in range(0, neuron_count):
    #     row = (str(weights[i][j]) for j in range(0, neuron_count))
    #     print(" ".join(row))
    return weights


def activation_rule(weights: List[List[int]], state: List[int], i: int) -> int:
    return sum(weights[i][j] * state[j] for j in
               range(0, len(weights)))


def update_state(weights: List[List[int]], state: List[int]) -> bool:
    """Updates the state, returning whether the state changed"""
    change_count = 0
    for i in range(0, len(state)):
        activation = activation_rule(weights, state, i)
        oldstate = state[i]
        state[i] = 1 if activation >= 0 else -1
        if state[i] != oldstate:
            change_count += 1
    return change_count > 0


def match(image_file: str, training_files: List[str]):
    all_file_contents = [Path(file).read_text().split("\n") for file in training_files]
    images, width, height = clean_images(all_file_contents)
    weights = train(images)
    image_data = clean_image(Path(image_file).read_text().split("\n"), width, height)
    assert(len(image_data) == height)
    assert(len(image_data[0]) == width)

    state = list(image_to_data(image_data))
    MAX_ITER = 10
    cur_iter = 0
    while cur_iter < MAX_ITER:
        cur_iter += 1
        changed = update_state(weights, state)
        if not changed:
            break

    final_image = [[cell == 1 for cell in state[y * width:(y + 1) * width]]
                   for y in range(0, height)]
    print("{} iterations".format(cur_iter))
    print("Final state:")
    print(image_to_text(final_image))

if len(sys.argv) <= 1:
    invalid_usage()
if sys.argv[1] == "train":
    all_file_contents = [Path(file).read_text().split("\n") for file in sys.argv[2:]]
    images, _, _ = clean_images(all_file_contents)
    train(images)
elif sys.argv[1] == "match":
    match(sys.argv[2], sys.argv[3:])
else:
    invalid_usage()
