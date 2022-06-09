import cv2
from PIL import Image
import numpy as np


def add_bounding_boxes(img, pred_cls, boxes, pred_score=None, thresh=0.35, rect_th=2, text_size=0.5, text_th=1,
                       color_box=(1, 1, 1), return_pil=True):
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
    # turn classification tensors to numpy
    pred_cls = pred_cls.numpy() # TODO: When have more classes - add mapping from int class to string

    if pred_score is not None:
        pred_score = pred_score.numpy()

    for i in range(len(boxes)):
        if pred_score is not None and pred_score[i] < thresh:
            continue
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]),
                       int(boxes[i][3])),
                      color=color_box, thickness=rect_th)
        print_text = f"fish{pred_cls[i]}"
        print_text = print_text if pred_score is None else print_text + ":{}".format(str(round(pred_score[i],3)))
        cv2.putText(img, print_text, (int(boxes[i][0] + rect_th), int(boxes[i][1])+20),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, color_box, thickness=text_th)

    # turn to PIL
    if return_pil:
        return Image.fromarray((img * 255).astype(np.uint8))

    return img

