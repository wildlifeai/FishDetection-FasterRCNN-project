import os
import argparse
import random
import pandas as pd
import cv2

DEFAULT_OUTPUT_FOLDER = ".\\data\\output"
DEFAULT_TRAINING_SIZE = 0.8
DEFAULT_VALIDATION_SIZE = 0.1


def extract_image_name(label_file_name) -> str:
    return label_file_name.split(".")[0] + ".jpg"


def write_file(dir_path, files_names, output_name):
    df = pd.DataFrame(columns=['label', 'x', 'y', 'h', 'w', 'image_name'])

    # currently we support only classification for one type there fore the label will be 1 (label zero means background)
    # The image location are in different measurement therefore multiplying by 1000

    for filename in files_names:
        image_name = extract_image_name(filename)
        im = cv2.imread(os.path.join(dir_path, "images", image_name))
        height, width, _ = im.shape
        labels_info = pd.read_csv(os.path.join(dir_path, "labels", filename), names=['label', 'x', 'y', 'h', 'w'], sep=' ')

        # The data is given in YOLO format therefore the x and y coordinates are in the middle
        y = list(map(lambda x: float(x), labels_info.y))
        x = list(map(lambda x: float(x), labels_info.x))
        w = list(map(lambda x: float(x), labels_info.w))
        h = list(map(lambda x: float(x), labels_info.h))
        label = list(map(lambda x: int(x) + 1, labels_info.label))
        y1 = [int((val * height) - ((w[i] / 2) * height)) for i, val in enumerate(y)]
        x1 = [int((val * width) - ((h[i] / 2) * width)) for i, val in enumerate(x)]
        x2 = [int((val * width) + ((h[i] / 2) * width)) for i, val in enumerate(x)]
        y2 = [int((val * height) + ((w[i] / 2) * height)) for i, val in enumerate(y)]
        df = df.append({'label': label, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'image_name': extract_image_name(image_name)}, ignore_index=True)

    df = df.drop(columns=['x', 'y', 'w', 'h'])

    df.to_csv(output_name, index=False)
    return df


def parse_data(dir_path, train_size, validation_size, output_path) -> None:
    """
    :param output_path:
    :param validation_size:
    :param train_size:
    :param dir_path: The directory path to the data folder
    """
    file_list = os.listdir(os.path.join(dir_path, "labels"))
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
    if not os.path.isdir(DEFAULT_OUTPUT_FOLDER):
        os.mkdir(DEFAULT_OUTPUT_FOLDER)

    parse_data(args.data_path, args.train_size, args.validation_size,
               DEFAULT_OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
