from torchvision.ops import batched_nms
import utils.transformers as T


def collate_fn(batch):
    """
    :param batch: The current batch
    :return: Zip the batch into a tuples
    """
    return tuple(zip(*batch))


def apply_mns(results, iou_thresh=0.3):
    """
    Returns the tensors after applying mns on the boxes and classifications.
    :param iou_thresh:
    :param results: results\targets object
    """
    mns_ind = batched_nms(results['boxes'], results['scores'],
                          results['labels'], iou_thresh)
    labels_filtered = results['labels'][mns_ind]
    scores_filtered = results['scores'][mns_ind]
    boxes_filtered = results['boxes'][mns_ind]

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
    return T.Compose(transforms)
