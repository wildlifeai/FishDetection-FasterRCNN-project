import os
import cv2
import wandb
import torchvision
import torch
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
from utils.general_utils import collate_fn, apply_mns, get_transform
from utils.plot_image_bounding_box import add_bounding_boxes
from torchvision.ops import box_iou

LOG_FREQUENCY = 1
NMS_THRESHOLD = 0.3
TEST_BATCH_SIZE = 10
SAVE_MODEL_FREQUENCY = 1
SHOULD_SAVE_MODEL = 1
VALIDATION_IOU_LOG = False


class FishDetectionModel:
    def __init__(self, args):
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")
        self.model = None

        if args.load_model:
            self.model = torch.load(args.load_model)

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
            self.model = self.build_model(5, False)

        # Creating data loaders
        dataset = SpyFishAotearoaDataset(self.args.data_path, "train.csv", get_transform(train=True))
        dataset_test = SpyFishAotearoaDataset(self.args.data_path, "validation.csv", get_transform(train=False))

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        self.model.to(device)
        verbose = self.args.verbose

        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay,
                                     betas=(0.09, 0.999))

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.learning_rate_size,
                                                       gamma=self.args.gamma)

        with wandb.init(config=vars(self.args)):
            for epoch in range(self.args.epochs):
                should_log = (epoch + 1) % LOG_FREQUENCY == 0
                print('Epoch {} of {}'.format(epoch + 1, self.args.epochs))
                avg_train_loss, avg_train_classifier, avg_train_rpn_box_reg, avg_train_objectness = self._train_one_epoch(
                    optimizer, data_loader, device, verbose)
                lr_scheduler.step()
                avg_val_loss, avg_val_classifier, avg_val_objectness, avg_val_rpn_box_reg = self._evaluate(
                    data_loader_test, should_log, device, verbose)

                if verbose:
                    print('\nLosses of epoch num {} are:'.format(epoch + 1))
                    print('Train loss: {}'.format(avg_train_loss))
                    print('Validation loss: {}'.format(avg_val_loss))

                wandb.log({"epoch": epoch + 1, "total_train_loss": avg_train_loss,
                           "total_validation_loss": avg_val_loss,
                           "avg_train_classifier": avg_train_classifier,
                           "avg_train_rpn_box_reg": avg_train_rpn_box_reg,
                           "avg_train_objectness": avg_train_objectness,
                           "avg_val_classifier": avg_val_classifier,
                           "avg_val_objectness": avg_val_objectness,
                           "avg_val_rpn_box_reg": avg_val_rpn_box_reg
                           })

                if epoch + 1 >= SHOULD_SAVE_MODEL and epoch % SAVE_MODEL_FREQUENCY == 0:
                    model_name = time.strftime("%Y%m%d-%H%M%S")
                    print('Saving model, epoch: {} name: {}'.format(epoch + 1, model_name))
                    torch.save(self.model, self.args.output_path + model_name)

            # Saving the model in the specified path
            model_name = time.strftime("%Y%m%d-%H%M%S")
            torch.save(self.model, self.args.output_path + model_name)

    def log_to_wb(self, images, targets, train=False, log_iou=True):
        """
        Logging predicted images and IOU to weights and biases
        :param train: If the log is connected to the train
        :param images: The images themselves
        :param targets: The real results of the images
        :param log_iou:  Boolean indicate whether to log the IOU data
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

                parsed_results[i] = apply_mns(parsed_results[i], NMS_THRESHOLD)

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

    def _train_one_epoch(self, optimizer, data_loader, device, verbose=True):
        self.model.train()
        avg_train_loss = 0
        avg_train_classifier = 0
        avg_train_objectness = 0
        avg_train_rpn_box_reg = 0

        for i, (images, targets, _) in enumerate(tqdm(data_loader, position=0, leave=True) if verbose else data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.empty_cache()
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            avg_train_loss += losses.cpu().item()

            with torch.no_grad():
                avg_train_classifier += loss_dict['loss_classifier'].cpu().item()
                avg_train_objectness += loss_dict['loss_objectness'].cpu().item()
                avg_train_rpn_box_reg += loss_dict['loss_rpn_box_reg'].cpu().item()

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

        return avg_train_loss, avg_train_classifier, avg_train_rpn_box_reg, avg_train_objectness

    def _evaluate(self, val_set, img_log, device, verbose=True):
        # In order to get the validation loss we need to use .train()
        self.model.train()
        avg_val_loss = 0
        avg_val_classifier = 0
        avg_val_objectness = 0
        avg_val_rpn_box_reg = 0

        with torch.no_grad():
            if verbose:
                print('\nStarting validation')

            for images, targets, _ in tqdm(val_set, position=0, leave=True) if verbose else val_set:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                torch.cuda.empty_cache()
                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                avg_val_loss += losses.cpu().item()

                avg_val_classifier += loss_dict['loss_classifier'].cpu().item()
                avg_val_objectness += loss_dict['loss_objectness'].cpu().item()
                avg_val_rpn_box_reg += loss_dict['loss_rpn_box_reg'].cpu().item()

                if img_log:
                    print("Logging validation images to weights and biases")
                    self.log_to_wb(images, targets, img_log)

        total_val_loss = avg_val_loss / len(val_set)
        avg_val_classifier /= len(val_set.dataset)
        avg_val_objectness /= len(val_set.dataset)
        avg_val_rpn_box_reg /= len(val_set.dataset)

        return total_val_loss, avg_val_classifier, avg_val_objectness, avg_val_rpn_box_reg

    def test(self, root_path, csv_path, output_path, class_names=None, cls_thresh=0.5, iou_thresh=0.4):
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

            for i, (images, targets, idx) in enumerate(tqdm(data_loader_test, position=0, leave=True)):
                images = list(image.to(device) for image in images)
                cur_index = idx[0]  # todo: check if index is necessary
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets][0]

                pred = self.model(images)

                pred = {
                    'boxes': pred[0]['boxes'].detach().cpu(),
                    'labels': pred[0]['labels'].cpu(),
                    'scores': pred[0]['scores'].detach().cpu()
                }

                iou = box_iou(targets['boxes'].cpu(), pred['boxes'])  # todo: check what boxes is all about
                max_iou = torch.max(iou, dim=1).values if iou.shape[1] != 0 else torch.Tensor([0])
                total_iou += torch.mean(max_iou)

                # log images here?
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
