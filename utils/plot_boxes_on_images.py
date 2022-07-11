import argparse
import os
import cv2
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
import utils.transformers as T
from utils.general_utils import collate_fn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot_image_bounding_box import add_bounding_boxes


# todo: check how to save our work in the computers

def print_images(img_path, file_name, should_save, output_path):
    """

    :param img_path:
    :param file_name:
    :param should_save:
    :param output_path:
    :return:
    """
    dataset = SpyFishAotearoaDataset(img_path, file_name, T.Compose([T.ToTensor()]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    for (images, targets, index) in tqdm(data_loader, position=0, leave=True):
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        num_boxes = len(targets[0]["boxes"])

        classes = torch.Tensor([0] * num_boxes)  # todo:change once we finish the classes

        log_images = images[0]
        log_images = log_images.cpu().numpy().transpose(1, 2, 0)
        log_images = cv2.cvtColor(log_images, cv2.COLOR_BGR2RGB)

        img = add_bounding_boxes(log_images, classes, targets[0]["boxes"])

        if should_save:
            image_name = dataset.get_image_name(index).split(".jpg")[0]
            img.save(os.path.join(output_path, image_name + "_with_labels.jpg"))
        else:
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="..\\data",
        help="path of the images to print")

    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=True,
        help="A boolean indicate if to save the files")

    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        default=".\\images_with_labels",
        help="A path to save the images with the boxes"
    )

    parser.add_argument(
        "-n",
        "--file_name",
        type=str,
        default="train.csv",
        help="The name of the csv file"
    )

    args = parser.parse_args()

    if args.save and not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    print_images(args.path, args.file_name, args.save, args.output_path)


if __name__ == '__main__':
    main()
