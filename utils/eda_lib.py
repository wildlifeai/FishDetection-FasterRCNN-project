"""
Function to generate and read EDA data on the data given to us
"""
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COL_WIDTH = 0.3


def dict_keys_to_np(keys):
    """
    Converts dict keys to a list of numbers
    :param keys: dict_keys
    :return: List of numbers
    """
    return np.array([int(y) for y in list(keys)])


def draw_bar_plot(x1, y1, title, x2=None, y2=None):
    """
    Presents a bar plot
    :param x2: optional
    :param y2: optional
    :param title: Graph title
    """
    plt.title(title)
    plt.bar(dict_keys_to_np(x1), y1, width=COL_WIDTH, color='g')
    if x2:
        plt.bar(dict_keys_to_np(x2) + 1.5*COL_WIDTH, y2, width=COL_WIDTH, color='r')
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

    # Classes
    count_dict = {}
    df["label"] = df["label"].apply(lambda x: count_types(x, count_dict))
    eda["n_class"] = count_dict

    # Size of boxes
    areas = df[['h', 'w']].apply(lambda x: get_areas(x[0], x[1]), axis=1)
    areas = areas.explode().astype(int)
    eda["box_area"] = areas.value_counts().to_dict()

    # Save to file
    with open(f'{file_path}.json', 'w') as outfile:
        json.dump(eda, outfile)


def present_eda(eda_data_train, eda_data_test=None):
    """
    Presents graphs that can explain the data.
    :param eda_data_test: Optional. If added the function will compare the eda data between train and test.
    :param eda_data_train: path to a json file that was created by create_eda
    """
    with open(eda_data_train, 'r') as f:
        data_train = json.load(f)

    if eda_data_test:
        with open(eda_data_test, 'r') as f:
            data_test = json.load(f)

    # Present graphs
    # # Number of objects in an image
    draw_bar_plot(data_train["n_objects"].keys(), data_train["n_objects"].values(),
                  "Number of objects in an image", data_train["n_objects"].keys(), data_train["n_objects"].values())

    # # Classes
    draw_bar_plot(data_train["n_class"].keys(), data_train["n_class"].values(), "Classes Distribution")

    # # Size of boxes
    x = [int(y) for y in list(data_train["box_area"].keys())]

    plt.title("Histogram of Box Area")
    plt.plot(x, data_train["box_area"].values(), color='g')
    plt.xticks(np.arange(min(x), max(x)+1, (max(x)-min(x))//5))
    plt.show()


if __name__ == "__main__":
    #create_eda("..\\data\\output\\train.csv", "..\\train_json")
    present_eda("../train_json.json", "../train_json.json")