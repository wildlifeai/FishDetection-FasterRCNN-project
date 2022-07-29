from torchvision.ops import batched_nms
import utils.transformers as T
import torch


def collate_fn(batch):
    """
    :param batch: The current batch
    :return: Zip the batch into a tuples
    """
    return tuple(zip(*batch))


def apply_mns(results, applyAll=False, iou_thresh=0.3):
    """
    Returns the tensors after applying mns on the boxes and classifications.
    :param applyAll:
    :param iou_thresh:
    :param results: results\targets object
    """
    results_scores = results['scores']
    results_boxes = results['boxes']
    results_labels = results['labels']

    results_labels_modify = results_labels if not applyAll else torch.zeros(len(results_boxes), dtype=torch.int32)

    mns_ind = batched_nms(results_boxes, results_scores,
                          results_labels_modify, iou_thresh)
    labels_filtered = results_labels[mns_ind]
    scores_filtered = results_scores[mns_ind]
    boxes_filtered = results_boxes[mns_ind]

    return {'boxes': boxes_filtered, 'labels': labels_filtered, 'scores': scores_filtered}


def get_transform(train):
    """
    Apply transformation to the images
    :param train: A boolean indicate if this dataset is train data set
    :return: The transformed images
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomGrayScale(0.1))
    return T.Compose(transforms)
