"""
Function to generate and read EDA data on the data given to us
"""
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


COL_WIDTH = 0.15
TRAIN_COL = 'g'
TEST_COL = 'r'
VAL_COL = 'b'


def dict_keys_to_np(keys):
    """
    Converts dict keys to a list of numbers
    :param keys: dict_keys
    :return: List of numbers
    """
    return np.array([int(y) for y in list(keys)])


def draw_bar_plot(x1, y1, title, x2=None, y2=None, x3=None, y3=None):
    """
    Presents a bar plot
    :param x2: optional
    :param y2: optional
    :param title: Graph title
    """
    plt.title(title)
    plt.bar(dict_keys_to_np(x1), y1, width=COL_WIDTH, color=TRAIN_COL, label="train set")
    if x2:
        plt.bar(dict_keys_to_np(x2) + 1*COL_WIDTH, y2, width=COL_WIDTH, color=TEST_COL, label="test set")
        plt.legend()
    if x3:
        plt.bar(dict_keys_to_np(x3) + 2 * COL_WIDTH, y3, width=COL_WIDTH, color=VAL_COL, label="validation "
                                                                                               "set")
        plt.legend()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.show()


def count_types(x, count_dict):
    """
    Gets a string that contains a list of types and updates a dictionary that count the amount of each type.
    """
    x = json.loads(x)
    for i in x:
        if i in count_dict.keys():
            count_dict[i] = count_dict[i] + 1
        else:
            count_dict[i] = 1
    return x


def get_areas(h_list, w_list) -> list:
    """
    Gets a list of hights and widths of squeres and returns a list of their areas
    """
    h_list = json.loads(h_list)
    w_list = json.loads(w_list)
    return [h * w for h, w in zip(h_list, w_list)]


def null_or_dict_val(dic, key):
    """
    :param dic: dictionary or null
    :param key: the key to search
    :return: returns null if dict doesn't exist and value otherwise
    """
    if dic:
        return dic[key]
    return None


def change_count_to_relative_dict(count_dict):
    """
    Changes count values to fraction values in a dictionary
    :param total: Total values in dict
    :param count_dict: Dictionary where the values are counters
    """
    total = sum(count_dict.values(), 0.0)
    return {k: v / total for k, v in count_dict.items()}


def create_eda(csv_path, file_path):
    """
    Read the csv file containing the information and saves the relevant information to a json object.
    :param csv_path: The full path to the file containing the train\test\validation data.
    :param file_path: The path to save the object to, without extension.
    """
    df = pd.read_csv(csv_path)

    # save relevant data
    eda = {}  # Dict to collect statistical data

    # Number of objects in an image - aggregate by image-name
    lens = df.applymap(lambda x: len(x))["label"]
    eda["n_objects"] = lens.value_counts().to_dict()
    eda["n_objects"] = change_count_to_relative_dict(eda["n_objects"])

    # Classes
    count_dict = {}
    df["label"] = df["label"].apply(lambda x: count_types(x, count_dict))
    eda["n_class"] = change_count_to_relative_dict(count_dict)

    # Size of boxes
    areas = df[['h', 'w']].apply(lambda x: get_areas(x[0], x[1]), axis=1)
    areas = areas.explode().astype(int)
    eda["box_area"] = list(areas)

    # Save to file
    with open(f'{file_path}.json', 'w') as outfile:
        json.dump(eda, outfile)


def present_eda(eda_data_train, eda_data_test=None, eda_data_val=None):
    """
    Presents graphs that can explain the data.
    :param eda_data_val: Optional.
    :param eda_data_test: Optional. If added the function will compare the eda data between train and test.
    :param eda_data_train: path to a json file that was created by create_eda
    """
    with open(eda_data_train, 'r') as f:
        data_train = json.load(f)

    # Load test data - if valid
    if eda_data_test:
        with open(eda_data_test, 'r') as f:
            data_test = json.load(f)

    # Load validation data - if valid
    if eda_data_test:
        with open(eda_data_val, 'r') as f:
            data_val = json.load(f)


    # Present graphs
    # # Number of objects in an image
    draw_bar_plot(data_train["n_objects"].keys(), data_train["n_objects"].values(),
                  "Number of objects in an image", null_or_dict_val(data_test, "n_objects").keys(),
                  null_or_dict_val(data_test, "n_objects").values(),
                  null_or_dict_val(data_val, "n_objects").keys(),
                  null_or_dict_val(data_val, "n_objects").values())

    # # Classes
    draw_bar_plot(data_train["n_class"].keys(), data_train["n_class"].values(), "Classes Distribution",
                  null_or_dict_val(data_test, "n_class").keys(),
                  null_or_dict_val(data_test, "n_class").values(),
                  null_or_dict_val(data_val, "n_class").keys(),
                  null_or_dict_val(data_val, "n_class").values()
                  )

    # # Size of boxes
    plt.hist(x=data_train["box_area"], bins='auto', color=TRAIN_COL,
             alpha=0.7, rwidth=0.85, label="train set")
    plt.hist(x=data_test["box_area"], bins='auto', color=TEST_COL,
             alpha=0.7, rwidth=0.85, label="test set")
    plt.hist(x=data_val["box_area"], bins='auto', color=VAL_COL,
             alpha=0.7, rwidth=0.85, label="validation set")
    plt.title("Histogram of Box Area")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    create_eda("..\\data\\output\\train.csv", "../eda_data/train_json")
    create_eda("..\\data\\output\\validation.csv", "../eda_data/validation_json")
    create_eda("..\\data\\output\\test.csv", "../eda_data/test_json")
    present_eda("../eda_data/train_json.json", "../eda_data/test_json.json",
                "../eda_data/validation_json.json")