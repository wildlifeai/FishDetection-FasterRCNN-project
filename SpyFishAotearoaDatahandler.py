import json
import os
import argparse
import random
import pandas as pd
from matplotlib import pyplot as plt

DEFAULT_OUTPUT_FOLDER = ".\\data\\output"
DEFAULT_TRAINING_SIZE = 0.9
DEFAULT_VALIDATION_SIZE = 0.1


def extract_image_name(label_file_name) -> str:
    return label_file_name.split(".")[0] + ".jpg"


def get_avg_area(h_list, w_list) -> list:
    return [h * w for h, w in zip(h_list, w_list)]


def write_file(dir_path, files_names, output_name):
    df = pd.DataFrame(columns=['label', 'x', 'y', 'h', 'w', 'image_name'])

    # currently we support only classification for one type there fore the label will be 1 (label zero means background)
    # The image location are in different measurement therefore multiplying by 1000
    for filename in files_names:
        img = pd.read_csv(os.path.join(dir_path, filename), names=['label', 'x', 'y', 'h', 'w'], sep=' ')
        df = df.append({'label': [1] * len(img.label), 'x': list(map(lambda x: float(x) * 1000, img.x)),
                        'y': list(map(lambda x: float(x) * 1000, img.y)),
                        'h': list(map(lambda x: float(x) * 1000, img.h)),
                        'w': list(map(lambda x: float(x) * 1000, img.w)),
                        'image_name': extract_image_name(filename)}, ignore_index=True)

    df.to_csv(output_name, index=False)
    return df


# todo complete this function
def save_eda(df, filepath) -> None:
    """
    Saves EDA on the data
    :param filepath: The file to save statistics on, including name but not extension
    :param df: Pandas dataframe
    """
    eda = {}  # Dict to collect statistical data

    # Number of objects in an image - aggregate by image-name
    lens = df.applymap(lambda x: len(x))["label"]
    eda["n_objects"] = lens.to_dict()
    print(f"Mean number of objects in an image {lens.mean()}")
    plt.title("Histogram of number of objects in an image")
    lens.hist()
    plt.show()

    # Classes
    classes = df["label"].explode()
    eda["n_class"] = classes.to_dict()
    print(f"classes distribution:\n {classes.value_counts()}")
    plt.title("Classes Distribution")
    classes.hist()
    plt.show()

    # Size of boxes
    areas = df[['h', 'w']].apply(lambda x: get_avg_area(x[0], x[1]), axis=1)
    areas = areas.explode()
    eda["box_area"] = areas.to_dict()
    print(f"Mean box area {areas.mean()}")
    plt.title("Histogram of Box Area")
    areas.hist()
    plt.show()

    # Save to file
    with open(f'{filepath}.json', 'w') as outfile:
        json.dump(eda, outfile)


def parse_data(dir_path, train_size, validation_size, output_path) -> None:
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

    # EDA
    save_eda(write_file(dir_path, file_list[:train], os.path.join(output_path, "train.csv")), "train_json")

    write_file(dir_path, file_list[:train], os.path.join(output_path, "train.csv"))
    write_file(dir_path, file_list[train:validation], os.path.join(output_path, "validation.csv"))
    write_file(dir_path, file_list[validation:], os.path.join(output_path, "test.csv"))

    # todo add a dictionary of number and class name


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

    parse_data(os.path.join(args.data_path, "labels"), args.train_size, args.validation_size,
               DEFAULT_OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
