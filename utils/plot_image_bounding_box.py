import cv2
import matplotlib.pyplot as plt #TODO: remove when done with simulation
import torch #TODO: remove when done with simulation
from tqdm import tqdm # TODO: remove when done with simulation
from PIL import Image

import utils.transformers as T  #TODO: remove when done with simulation
from utils.general_utils import collate_fn #TODO: remove when done with simulation


from SpyFishAotearoaDataset import SpyFishAotearoaDataset #TODO: remove when done with simulation
import numpy as np


def get_transform(train):  #TODO: remove when done with simulation
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def add_bounding_boxes(img, pred_cls, pred_score, boxes, rect_th=2, text_size=1, text_th=1,
                       color_box=(255, 0, 0)):
    """
    Returns the image with boxes and labels as PIL
    :param img: The image
    :param pred_cls: Array of classes (as numbers)
    :param pred_score: array of probabilities
    :param boxes: Array of boxes
    :param rect_th: thickness of the rectangle
    :param text_size: Text size
    :param text_th: thickness of the text
    :param color_box: Box and text color
    :return: Returns the image with boxes and labels as PIL
    """
    img = img.numpy().T.swapaxes(0, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # turn classification tensors to numpy
    pred_cls = pred_cls.numpy() # TODO: When have more classes - add mapping from int class to string
    pred_score = pred_score.numpy()

    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][0] + boxes[i][3]),
                       int(boxes[i][1] + boxes[i][2])),
                      color=color_box, thickness=rect_th)
        cv2.putText(img, "fish" + ": " + str(round(pred_score[i], 3)) + "%",
                    (int(boxes[i][0]), int(boxes[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, color_box, thickness=text_th)  # TODO: add real class label

    # turn to PIL
    return Image.fromarray((img * 255).astype(np.uint8))

if __name__ == '__main__':
    # TODO: remove when done with simulation
    dataset_test = SpyFishAotearoaDataset("../first_image/", get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    iterable = tqdm(data_loader_test, position=0, leave=True)
    for images, targets in iterable:
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        img = add_bounding_boxes(images[0], torch.Tensor([4]), torch.Tensor([80]),  targets[0]["boxes"])

        # img = images[0].numpy().T.swapaxes(0,1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # boxes = targets[0]["boxes"]

        # for i in range(len(boxes)):
        #     cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
        #                   (int(boxes[i][0] + boxes[i][3]),
        #                   int(boxes[i][1] + boxes[i][2])),
        #                   color=(0, 255, 0), thickness=2)
        #     cv2.putText(img, "fish" + ": " + str(50), (int(boxes[i][0]), int(boxes[i][1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=1) # TODO: add real label
        #     # pred_cls[i]


        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        break

    #boxes, pred_cls, pred_score = get_prediction(img_path, confidence)
    # img = cv2.imread("../first_image/Images/c9a75cb0-80b1-4298-b16c-992f43852ff3.jpeg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # print(len(boxes))
    # # for i in range(len(boxes)):
    # #     cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    # #     cv2.putText(img, pred_cls[i] + ": " + str(round(pred_score[i], 3)), boxes[i][0],
    # #                 cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()