"""
Object for evaluation of a model
"""
import torch


class ModelTestEval:
    def __init__(self, model_path, test_dataset=None):
        self.model = torch.load(model_path)
        self.dataset = test_dataset

    def roc_curve(self, iou_range=None, iou_step=0.1):
        """
        Plots a ROC curve of IOU threshold vs. recall.
        """
        pass
        # todo: complete the function
        # if iou_range is None:
        #     iou_range = [0.1, 0.7]
        #
        # truth = None
        # for cur_iou in range(iou_range[0], iou_range[1], iou_step):
        #     results = self.model.predict(self.dataset.image, "cpu", ["fish"], iou_thresh=iou_step)
        #     for images, targets in iterable:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #
        #         loss_dict = self.model(images, targets)
        #
        #         losses = sum(loss for loss in loss_dict.values())
        #         avg_val_loss += losses.item()

