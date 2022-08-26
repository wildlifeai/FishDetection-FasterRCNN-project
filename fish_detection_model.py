import os
import cv2
import numpy as np
import wandb
import torchvision
import torch
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
from utils.general_utils import collate_fn, apply_mns, get_transform
from utils.mlp_network import MLPHead
from utils.plot_image_bounding_box import add_bounding_boxes
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CHECKPOINT_FREQUENCY = 20
LOG_FREQUENCY = 10
NMS_THRESHOLD = 0.3
TEST_BATCH_SIZE = 8
SAVE_MODEL_FREQUENCY = 20
SHOULD_SAVE_MODEL = 121
VALIDATION_IOU_LOG = False
MIN_SIZE = 480
MAX_SIZE = 1920


class FishDetectionModel:
    def __init__(self, args):
        self.model = None

        if args.load_model:
            self.model = torch.load(args.load_model)

        self.args = args

    def train(self, config=None):
        if config is None:
            config = vars(self.args)

        with wandb.init(config=config):
            config = wandb.config
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print(f"using {device} as device")

            # Creating data loaders
            dataset_with_style = SpyFishAotearoaDataset(self.args.data_path, "style_train.csv", get_transform(train=True))
            dataset_test = SpyFishAotearoaDataset(self.args.data_path, "validation.csv", get_transform(train=False))

            data_loader_style = torch.utils.data.DataLoader(
                dataset_with_style, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

            # create a model if it doesn't have path
            if self.model is None:
                self.model = self.build_model(5, False)

            verbose = self.args.verbose

            params = [p for p in self.model.parameters() if p.requires_grad]

            optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay,
                                         betas=(0.09, 0.999))

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_size,
                                                           gamma=config.gamma)

            epoch_checkpoint = 0
            if self.args.load_checkpoint:
                checkpoint = torch.load(self.args.checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_checkpoint = checkpoint['epoch']

            self.model.to(device)

            for epoch in range(epoch_checkpoint, config.epochs):
                should_log = (epoch + 1) % LOG_FREQUENCY == 0
                print('Epoch {} of {}'.format(epoch + 1, self.args.epochs))

                avg_train_loss, avg_train_classifier, avg_train_rpn_box_reg, avg_train_objectness, avg_train_box_reg = \
                    self._train_one_epoch(optimizer, data_loader_style, device, verbose)
                lr_scheduler.step()
                avg_val_loss, avg_val_classifier, avg_val_objectness, avg_val_rpn_box_reg, avg_val_box_reg = \
                    self._evaluate(data_loader_test, should_log, device, verbose)

                if verbose:
                    print('\nLosses of epoch num {} are:'.format(epoch + 1))
                    print('Train loss: {}'.format(avg_train_loss))
                    print('Validation loss: {}'.format(avg_val_loss))

                wandb.log({"epoch": epoch + 1, "total_train_loss": avg_train_loss,
                           "total_validation_loss": avg_val_loss,
                           "avg_train_classifier": avg_train_classifier,
                           "avg_train_rpn_box_reg": avg_train_rpn_box_reg,
                           "avg_train_objectness": avg_train_objectness,
                           "avg_train_box_reg": avg_train_box_reg,
                           "avg_val_classifier": avg_val_classifier,
                           "avg_val_objectness": avg_val_objectness,
                           "avg_val_rpn_box_reg": avg_val_rpn_box_reg,
                           "avg_val_box_reg":avg_val_box_reg
                           })

                if epoch + 1 >= SHOULD_SAVE_MODEL and epoch % SAVE_MODEL_FREQUENCY == 0:
                    cur_name = time.strftime("%Y%m%d-%H%M%S")
                    print('Saving model, epoch: {} name: {}'.format(epoch + 1, cur_name))
                    torch.save(self.model, self.args.output_path + cur_name + ".model")

                if epoch + 1 >= SHOULD_SAVE_MODEL and epoch % CHECKPOINT_FREQUENCY == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                    }, os.path.join(self.args.checkpoint_path, cur_name + ".pt"))

            # Saving the last model
            cur_name = time.strftime("%Y%m%d-%H%M%S")
            name = cur_name + "_TrainedModel" + ".model"
            print('Saving last model,name: {}'.format(name))
            torch.save(self.model, self.args.output_path + name)
            self.model = None

    def _train_one_epoch(self, optimizer, data_loader, device, add_classification_weight=False, verbose=True):
        self.model.train()
        avg_train_loss = 0
        avg_train_classifier = 0
        avg_train_objectness = 0
        avg_train_rpn_box_reg = 0
        avg_train_box_reg = 0

        for i, (images, targets, _) in enumerate(tqdm(data_loader, position=0, leave=True) if verbose else data_loader):
            avg_train_loss, images, loss_dict, losses, targets = self.predict_images_and_calculate_loss(avg_train_loss,
                                                                                                        device, images,
                                                                                                        targets, add_classification_weight)

            with torch.no_grad():
                avg_train_classifier += loss_dict['loss_classifier'].cpu().item()
                avg_train_objectness += loss_dict['loss_objectness'].cpu().item()
                avg_train_rpn_box_reg += loss_dict['loss_rpn_box_reg'].cpu().item()
                avg_train_box_reg += loss_dict['loss_box_reg'].cpu().item()

                # Because of shuffle in the train data set we always take the first image of each epoch
                img_log = i == 0
                if img_log:
                    print("Logging train image to weights and biases")
                    self.log_to_wb(images[:1], targets[:1], train=True)

            # Zero any old/existing gradients on the model's parameters
            optimizer.zero_grad()
            # Compute gradients for each parameter based on the current loss calculation
            losses.backward()
            # Update model parameters from gradients: param -= learning_rate * param.grad
            optimizer.step()

        avg_train_loss /= len(data_loader.dataset)
        avg_train_objectness /= len(data_loader.dataset)
        avg_train_rpn_box_reg /= len(data_loader.dataset)
        avg_train_classifier /= len(data_loader.dataset)

        return avg_train_loss, avg_train_classifier, avg_train_rpn_box_reg, avg_train_objectness, avg_train_box_reg

    def _evaluate(self, val_set, img_log, device, verbose=True):
        # In order to get the validation loss we need to use .train()
        self.model.train()
        avg_val_loss = 0
        avg_val_classifier = 0
        avg_val_objectness = 0
        avg_val_rpn_box_reg = 0
        avg_val_box_reg = 0

        with torch.no_grad():
            if verbose:
                print('\nStarting validation')

            for images, targets, _ in tqdm(val_set, position=0, leave=True) if verbose else val_set:
                avg_val_loss, images, loss_dict, losses, targets = self.predict_images_and_calculate_loss(
                    avg_val_loss,
                    device, images,
                    targets)

                avg_val_classifier += loss_dict['loss_classifier'].cpu().item()
                avg_val_objectness += loss_dict['loss_objectness'].cpu().item()
                avg_val_rpn_box_reg += loss_dict['loss_rpn_box_reg'].cpu().item()
                avg_val_box_reg += loss_dict['loss_box_reg'].cpu().item()

                if img_log:
                    print("Logging validation images to weights and biases")
                    self.log_to_wb(images, targets)

        total_val_loss = avg_val_loss / len(val_set)
        avg_val_classifier /= len(val_set.dataset)
        avg_val_objectness /= len(val_set.dataset)
        avg_val_rpn_box_reg /= len(val_set.dataset)

        return total_val_loss, avg_val_classifier, avg_val_objectness, avg_val_rpn_box_reg, avg_val_box_reg

    def test(self, root_path, output_path, nms_thresh=0.3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"using {device} as device")

        # building dataloader
        dataset_test = SpyFishAotearoaDataset(root_path, "test.csv", get_transform(train=False))

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            print('Starting to test over the data set\n')
            total_iou = 0
            false_positive_classification = 0
            miss_detections_rate = 0

            # For calculating mAP
            metric = MeanAveragePrecision()

            for i, (images, targets, idx) in enumerate(tqdm(data_loader_test, position=0, leave=True)):
                images = list(image.to(device) for image in images)
                cur_index = idx[0]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                pred = self.model(images)

                pred = {
                    'boxes': pred[0]['boxes'].detach().cpu(),
                    'labels': pred[0]['labels'].cpu(),
                    'scores': pred[0]['scores'].detach().cpu()
                }

                targets = {
                    'boxes': targets[0]['boxes'].detach().cpu(),
                    'labels': targets[0]['labels'].cpu(),
                }

                # apply NMS on prediction
                pred = apply_mns(pred, nms_thresh)
                pred = apply_mns(pred, applyAll=True, iou_thresh=0.8)

                # Update mAp
                metric.update([pred], [targets])

                # Update IOU
                iou = box_iou(targets['boxes'].cpu(), pred['boxes'].cpu())

                max_iou = torch.max(iou, dim=1).values if iou.shape[1] != 0 else torch.Tensor([0])
                total_iou += torch.mean(max_iou)

                len_targets = len(targets['boxes'].cpu())

                # if the shape of the iou is zero then it means that the model didn't predict nothing
                # so we need to ignore these cases
                no_boxes_predicted = iou.shape[1] != 0

                if no_boxes_predicted and len_targets != 0:
                    # Calculate miss classification - counting the number of IOU less than 0.3
                    miss_detections_rate += 0 if len_targets == 0\
                        else torch.count_nonzero(max_iou < 0.3) / len_targets

                    # false_positive_classification
                    matching_index_iou = torch.argmax(iou, dim=1)
                    count_miss_class = np.count_nonzero(targets['labels'] != pred['labels'][matching_index_iou])
                    false_positive_classification += count_miss_class / len_targets

                # log images here
                image_to_log = images[0].cpu().numpy().transpose(1, 2, 0)
                image_to_log = cv2.cvtColor(image_to_log, cv2.COLOR_BGR2RGB)

                image_to_log = add_bounding_boxes(image_to_log, pred['labels'],
                                                  pred['boxes'],
                                                  pred_score=pred['scores'],
                                                  color_box=(1, 1, 1),
                                                  thresh=0.001,
                                                  return_pil=True
                                                  )
                image_to_log.save(os.path.join(output_path, "test_" + dataset_test.get_image_name(cur_index)))

            print("Average IOU all over the images: {}".format(total_iou / len(data_loader_test)))
            print("Average miss classification all over the images: {}".format(false_positive_classification / len(data_loader_test)))
            print("Average miss detection all over the images: {}".format(miss_detections_rate / len(data_loader_test)))
            print("mAP score over all test images:{}".format(metric.compute()))

    def predict_images_and_calculate_loss(self, avg_loss, device, images, targets,
                                          add_classification_weight=False):
        """
        Predicting the images and calculate the loss for the model
        :param add_classification_weight: If true, gives more weight to the classifier loss.
        :param avg_loss: The current average train loss until now
        :param device: The deice to work on
        :param images: The images to predict
        :param targets: The true labels
        """

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.empty_cache()
        loss_dict = self.model(images, targets)

        if add_classification_weight:
            loss_dict['loss_classifier'] *= 1.01

        losses = sum(loss for loss in loss_dict.values())
        avg_loss += losses.cpu().item()

        return avg_loss, images, loss_dict, losses, targets

    def build_model(self, num_classes=2, pretrained=True):
        """
        Building a new pretrained model of faster rcnn
        :param pretrained: Define if the model is pretrained
        :param num_classes: The number of classes to predict
        :return: faster-rcnn model
        """

        if self.args.dropout is not None:
            print("Using generated MLP box head")
            box_head = MLPHead(dropout=self.args.dropout)
        else:
            print("Using default box head")
            box_head = None

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, box_head=box_head,
                                                                     min_size=MIN_SIZE, max_size=MAX_SIZE)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def log_to_wb(self, images, targets, train=False):
        """
        Logging predicted images and IOU to weights and biases
        :param train: If the log is connected to the train
        :param images: The images themselves
        :param targets: The real results of the images
        :return: batch IOU sum and number of boxes if log_iou is true else None
        """
        with torch.no_grad():
            self.model.eval()  # using evaluation mode in order to get the classifications
            results = self.model(images)
            parsed_results = []
            all_images = []

            for i, img in enumerate(images):
                parsed_results.append(
                    {
                        'labels': results[i]['labels'].cpu(),
                        'boxes': results[i]['boxes'].cpu(),
                        'scores': results[i]['scores'].cpu()
                    }
                )

                parsed_results[i] = apply_mns(parsed_results[i], iou_thresh=NMS_THRESHOLD)
                parsed_results[i] = apply_mns(parsed_results[i], applyAll=True, iou_thresh=0.8)

                # Adding bounding boxes of the prediction
                image_to_log = img.cpu().numpy().transpose(1, 2, 0)
                image_to_log = cv2.cvtColor(image_to_log, cv2.COLOR_BGR2RGB)

                image_to_log = add_bounding_boxes(image_to_log, parsed_results[i]['labels'],
                                                  parsed_results[i]['boxes'],
                                                  pred_score=parsed_results[i]['scores'],
                                                  color_box=(1, 1, 1),
                                                  thresh=0.001,
                                                  return_pil=False
                                                  )

                # Adding bounding box of the real targets
                image_to_log = add_bounding_boxes(image_to_log, targets[i]['labels'].cpu(),
                                                  targets[i]['boxes'].cpu(),
                                                  color_box=(0, 0, 0),
                                                  thresh=0.001
                                                  )

                all_images.append(image_to_log)

            self.model.train()

            log = "classifications_images_train_set" if train else "classifications_images_validation_set"
            wandb.log({log: [wandb.Image(image) for image in all_images]})
