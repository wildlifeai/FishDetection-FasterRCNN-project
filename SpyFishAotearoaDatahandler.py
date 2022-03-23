import argparse
import pandas as pd


def change_labels(df):
    df['label'].replace(['Blue cod', 'Bait'], [1, 2], inplace=True)


def merge_rows(df):
    grouped_df = df.groupby('subject_ids').agg({'label': lambda x: list(x), 'x': lambda x: list(x),
                                                'y': lambda x: list(x), 'h': lambda x: list(x),
                                                'w': lambda x: list(x), 'https_location': lambda x: x}).reset_index()

    return grouped_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--csv_file_path",
        type=str,
        help="The path of the csv file")

    args = parser.parse_args()
    spyfish_df = pd.read_csv(args.csv_file_path)
    change_labels(spyfish_df)
    spyfish_df = merge_rows(spyfish_df)

    # changing the file itself
    spyfish_df.to_csv(args.csv_file_path)


if __name__ == '__main__':
    main()
