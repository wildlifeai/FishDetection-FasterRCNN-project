import os
from PIL.Image import Image
import wandb
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
import utils.transformers as T
from utils.general_utils import collate_fn
from utils.plot_image_bounding_box import add_bounding_boxes
from torchvision.ops import box_iou, batched_nms

LOG_TRAIN_FREQ = 10
VALIDATION_IOU_LOG = False


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


class FishDetectionModel:
    def __init__(self, args):
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")
        self.model = None
        self.args = args

    def build_model(self, num_classes=2, pretrained=True):
        """
        Building a new pretrained model of faster rcnn
        :param pretrained: Define if the model is pretrained
        :param num_classes: The number of classes to predict
        :return: faster-rcnn model
        """
        if self.args.model_type:
            # Choose to use a different model than the default
            model = self.args.model_type(pretrained=pretrained)
        else:
            # Using default.
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def train(self):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"using {device} as device")

        # create a model if it doesn't have path
        if self.model is None:
            self.model = self.build_model(2, False)

        # Creating data loaders
        dataset = SpyFishAotearoaDataset(self.args.data_path, "train.csv", get_transform(train=True))
        dataset_test = SpyFishAotearoaDataset(self.args.data_path, "validation.csv", get_transform(train=False))

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.model.to(device)
        verbose = self.args.verbose

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.args.learning_rate, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.learning_rate_size,
                                                       gamma=self.args.gamma)

        with wandb.init(config=vars(self.args)):
            for epoch in range(self.args.epochs):
                should_log = epoch % 1 == 0
                print('Epoch {} of {}'.format(epoch + 1, self.args.epochs))
                avg_train_loss = self.train_one_epoch(optimizer, data_loader, device, verbose)
                lr_scheduler.step()
                avg_val_loss, avg_iou = self.evaluate(data_loader_test, should_log, device, verbose)

                if verbose:
                    print(f'\nLosses of epoch num {epoch + 1} are:')
                    print('Train loss: {}'.format(avg_train_loss))
                    print('Validation loss: {}'.format(avg_val_loss))
                    if should_log:
                        print('IOU average : {}'.format(avg_iou))

                if should_log:
                    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "validation_loss": avg_val_loss,
                               'iou_average': avg_iou})
                else:
                    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "validation_loss": avg_val_loss})

            # Saving the model in the specified path
            torch.save(self.model, self.args.output_path)

    def log_to_wb(self, images, targets, log_img=True, log_iou=True):
        """
        Logging predicted images and IOU to weights and biases
        :param images: The images themselves
        :param targets: The real results of the images
        :param log_img: Boolean indicate whether to log the images
        :param log_iou:  Boolean indicate whether to log the IOU data
        :return: batch IOU sum and number of boxes if log_iou is true else None
        """
        with torch.no_grad():
            img_boxes = []
            batch_iou_sum = 0
            n_boxes = 0

            self.model.eval()
            results = self.model(images)

            for i, img in enumerate(images):
                if log_img:
                    img_boxes.append(add_bounding_boxes(img.cpu(), results[i]['labels'].cpu(),
                                                        results[i]['scores'].cpu(), results[i]['boxes'].cpu(),
                                                        thresh=0.20))

                if log_iou:
                    batch_iou_sum += box_iou(targets[i]["boxes"].cpu(), results[i]["boxes"].cpu())
                    n_boxes += results[i]["boxes"].shape[0]

            self.model.train()

            if log_img:
                wandb.log({"classifications_images": [wandb.Image(image) for image in img_boxes]})

            if log_iou:
                return batch_iou_sum, n_boxes

    def train_one_epoch(self, optimizer, data_loader, device, verbose=True):
        self.model.train()
        avg_train_loss = 0

        for i, (images, targets) in enumerate(tqdm(data_loader, position=0, leave=True) if verbose else data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.empty_cache()
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            avg_train_loss += losses.item()

            # Zero any old/existing gradients on the model's parameters
            optimizer.zero_grad()
            # Compute gradients for each parameter based on the current loss calculation
            losses.backward()
            # Update model parameters from gradients: param -= learning_rate * param.grad
            optimizer.step()

            # Logging train images to w&b after model prediction
            if i % LOG_TRAIN_FREQ == 0:
                # TODO: log predictions and targets
                self.log_to_wb(images, targets, log_img=True, log_iou=False)

        return avg_train_loss / len(data_loader.dataset)

    def evaluate(self, val_set, img_log, device, verbose=True):
        # In order to get the validation loss we need to use .train()
        self.model.train()
        avg_iou = 0
        boxes_num = 0  # How many boxes the model found
        avg_val_loss = 0

        with torch.no_grad():
            if verbose:
                print('\nStarting validation')

            for images, targets in tqdm(val_set, position=0, leave=True) if verbose else val_set:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                avg_val_loss += losses.item()

                if VALIDATION_IOU_LOG:
                    iou_results = self.log_to_wb(images, targets, img_log)
                    avg_iou += iou_results[0].sum()
                    boxes_num += iou_results[1]

        return avg_val_loss / len(val_set), 0 if img_log else avg_val_loss / len(val_set)

    def predict(self, img_path, device, class_names, cls_thresh=0.5, iou_thresh=0.4):
        self.model.eval()
        with torch.no_grad():
            img = Image.open(img_path)
            transform = T.Compose([T.ToTensor()])
            img = transform(img).to(device)
            pred = self.model([img])

            # TODO: check mns code using GPU
            mns_ind = batched_nms(pred[0]['boxes'].detach().cpu(), pred[0]['scores'].detach().cpu(),
                                  pred[0]['labels'].cpu(), iou_thresh)
            labels_filtered = pred[0]['labels'].cpu()[mns_ind, :]
            scores_filtered = pred[0]['scores'].detach().cpu()[mns_ind, :]
            boxes_filtered = pred[0]['labels'].cpu()[mns_ind, :]

            pred_class = [class_names[i] for i in list(labels_filtered.numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(boxes_filtered.numpy())]
            pred_score = list(scores_filtered.numpy())

            pred_t = [pred_score.index(x) for x in pred_score if x > cls_thresh][-1]

            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]
            pred_score = pred_score[:pred_t + 1]

            images = add_bounding_boxes(img, pred_class, pred_score, pred_boxes, thresh=iou_thresh)
            return images, pred_boxes, pred_class, pred_score
