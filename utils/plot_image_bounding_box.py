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


def add_bounding_boxes(img, pred_cls, pred_score, boxes, thresh=0.35, rect_th=2, text_size=0.8, text_th=1,
                       color_box=(255, 255, 255)):
    """
    Returns the image with boxes and labels as PIL
    :param thresh:
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
        if pred_score[i] < thresh:
            continue
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]),
                       int(boxes[i][3])),
                      color=color_box, thickness=rect_th)
        cv2.putText(img, "fish" + ":" + str(round(pred_score[i], 3)),
                    (int(boxes[i][0]), int(boxes[i][1])),
                    cv2.FONT_HERSHEY_DUPLEX, text_size, color_box, thickness=text_th)  # TODO: add real class label

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

        img = add_bounding_boxes(images[0], torch.Tensor([4]), torch.Tensor([0.8]),  targets[0]["boxes"])

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        break

