#!/usr/bin/env python3

import argparse
import itertools
from pathlib import Path
import re
import sys
from typing import List, Iterable


class Image:
    def __init__(self, pbm: str):
        pbm = re.sub(r"#\n", "", pbm).split()
        assert(pbm[0] == "P1")
        self.width = int(pbm[1])
        self.height = int(pbm[2])
        pbm = pbm[3:]
        self.raster = [[bool(int(pixel)) for pixel in pbm[row:row + self.width]] for row in range(0, len(pbm), self.width)]

    def resize(self, width: int, height: int):
        assert(self.width <= width and self.height <= height)
        self.raster = ([[False] * width for _ in range((height - self.height) // 2)]) + \
            [[False] * ((width - self.width) // 2) + row + [False] * -((self.width - width) // 2) for row in self.raster] + \
            ([[False] * width for _ in range(-(self.height - height) // 2)])
        self.width = width
        self.height = height

    @property
    def data(self) -> Iterable[int]:
        data = []
        for row in self.raster:
            for value in row:
                data.append(1 if value else -1)
        return data

    def data_to_pbm(data: List[int], width: int, height: int) -> str:
        pbm = "P1\n{} {}\n".format(width, height) + \
            "\n".join([" ".join(["1" if pixel == 1 else "0" for pixel in data[row:row + width]]) for row in range(0, len(data), width)])
        return pbm


def train(images: List[Image]) -> List[List[int]]:
    assert(images)
    image_height = images[0].height
    image_width = images[0].width
    neuron_count = image_width * image_height
    weights = [list(itertools.repeat(0, neuron_count))
               for _ in range(0, neuron_count)]
    img_nums = [list(image.data) for image in images]
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


def match(image_file: str, model_file: str):
    image = Image(Path(image_file).read_text())
    model = model_file.read().split("\n")
    weights = [list(map(int, row.split())) for row in model][:image.width * image.height]

    state = image.data
    MAX_ITER = 10
    for cur_iter in range(MAX_ITER):
        print("State", cur_iter, file=sys.stderr)
        print(Image.data_to_pbm(state, image.width, image.height), file=sys.stderr)
        cur_iter += 1
        changed = update_state(weights, state)
        if not changed:
            break

    print("{} iterations".format(cur_iter), file=sys.stderr)
    print(Image.data_to_pbm(state, image.width, image.height))


def main(args: List[str]):
    cmd_parser = argparse.ArgumentParser(description="""
        Implementation of a Hopfield network.
        """)
    subparsers = cmd_parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="""Train the
        Hopsfield network on the specified files, which are expected to be plain
        PBM files. The generated model is printed to standard output.""")
    train_parser.add_argument("training_files", nargs="+")

    match_parser = subparsers.add_parser("match", help="""
        Find the best match, as determined from the model taken from standard
        input, for the specified file (a plain PBM file). The match is printed
        to standard output.""")
    match_parser.add_argument("test_file")
    match_parser.add_argument("model_file", nargs="?", type=argparse.FileType("r"), default=sys.stdin)

    result = cmd_parser.parse_args()

    if result.command == "train":
        images = [Image(Path(f).read_text()) for f in result.training_files]
        pixels = [image.raster for image in images]
        width = max(map(len, itertools.chain(*pixels)))
        height = max(map(len, pixels))
        for image in images:
            image.resize(width, height)
        model = train(images)
        print("\n".join(" ".join(map(str, row)) for row in model))
    elif result.command == "match":
        match(result.test_file, result.model_file)


if __name__ == "__main__":
    main(sys.argv)
