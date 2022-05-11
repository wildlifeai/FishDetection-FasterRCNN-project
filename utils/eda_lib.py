"""
Function to generate and read EDA data on the data given to us
"""
import json

import pandas as pd
from matplotlib import pyplot as plt


def get_avg_area(h_list, w_list) -> list:
    return [h * w for h, w in zip(h_list, w_list)]


def create_eda(df_path, file_path):
    """
    Read the csv file containing the information and saves the relevant information to a json object.
    :param df_path: The full path to the file containing the train\test\validation data.
    :param file_path: The path to save the object to, without extension.
    """
    df = pd.read_csv(df_path)

    # save relevant data
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
    with open(f'{file_path}.json', 'w') as outfile:
        json.dump(eda, outfile)

if __name__ == "__main__":
    create_eda()