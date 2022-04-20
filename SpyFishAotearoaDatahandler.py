import os
import argparse
import random
import pandas as pd

DEFAULT_OUTPUT_FOLDER = ".\\data\\output"
DEFAULT_TRAINING_SIZE = 0.9
DEFAULT_VALIDATION_SIZE = 0.1


def get_image_name(label_file_name):
    return label_file_name.split(".")[0] + ".jpg"


def write_file(dir_path, files_names, output_name):
    df = pd.DataFrame(columns=['label', 'x', 'y', 'h', 'w', 'image_name', 'box_size'])

    for filename in files_names:
        with open(os.path.join(dir_path, filename)) as f:
            content = f.read().splitlines()
            for line in content:
                splitted_line = line.split(" ")
                df = df.append({'label': splitted_line[0], 'x': float(splitted_line[1]), 'y': float(splitted_line[2]),
                                'h': float(splitted_line[3]), 'w': float(splitted_line[4]),
                                'box_size': float(splitted_line[3]) * float(splitted_line[4]),
                                'image_name': get_image_name(filename)}, ignore_index=True)

    df.to_csv(output_name)


def parse_label_directory(dir_path, train_size, validation_size, output_path) -> None:
    """
    :param output_path:
    :param validation_size:
    :param train_size:
    :param dir_path: The directory path to the data folder
    """
    file_list = os.listdir(dir_path)
    random.shuffle(file_list)

    file_list_len = len(file_list)
    train = int(file_list_len * train_size)
    validation = train + int(file_list_len * validation_size) + 1

    write_file(dir_path, file_list[:train], os.path.join(output_path, "train.csv"))
    write_file(dir_path, file_list[train:validation], os.path.join(output_path, "validation.csv"))
    write_file(dir_path, file_list[validation:], os.path.join(output_path, "test.csv"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--data_path",
        type=str,
        default=".\\data",
        help="The path of the data directory")

    parser.add_argument(
        "-t",
        "--train_size",
        type=float,
        default=DEFAULT_TRAINING_SIZE,
        help="The percentage size of the training"
    )

    parser.add_argument(
        "-v",
        "--validation_size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="The percentage size of validation"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="The output path of the '.csv' files",
        default=DEFAULT_OUTPUT_FOLDER
    )

    args = parser.parse_args()

    # generating the default output folder if not exists
    if args.output_path == DEFAULT_OUTPUT_FOLDER and not os.path.isdir(DEFAULT_OUTPUT_FOLDER):
        os.mkdir(DEFAULT_OUTPUT_FOLDER)

    parse_label_directory(os.path.join(args.data_path, "labels"), args.train_size, args.validation_size,
                          args.output_path)


if __name__ == '__main__':
    main()
