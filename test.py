import os
import argparse
import torch
from fish_detection_model import FishDetectionModel

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lm",
        "--load_model",
        type=str,
        help="path of the trained model")

    parser.add_argument(
        "-pt",
        "--path_data_csv",
        type=str,
        help="path of the csv of data")

    parser.add_argument(
        "-pd",
        "--path_data",
        type=str,
        help="path of the data")

    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        default='./test_output',
        help="path of the data")

    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    model = FishDetectionModel(args)
    model.test(args.path_data, args.path_data_csv, args.output_path)


if __name__ == '__main__':
    main()
