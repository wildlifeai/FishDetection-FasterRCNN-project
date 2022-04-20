import os
from PIL.Image import Image
from utils.plot_image_bounding_box import add_bounding_boxes
import wandb
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
import utils.transformers as T
from utils.general_utils import collate_fn
from torchvision.ops import box_iou
from utils.plot_image_bounding_box import add_bounding_boxes


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class FishDetectionModel:
    def __init__(self, args):
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")

        self.model = None
        if args.dry_run:
            os.environ['WANB_MODE'] = 'dryrun'

        if args.load_model:
            # todo load model from path
            # self.model = load.model...
            pass

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
        # todo: saving the image and the box in w&b?
        # todo: understand the flow of retrieving the data
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        print(f"using {device} as device")

        if self.model is None:
            self.model = self.build_model()

        # todo: create a data loader for validation

        # Creating data loader
        dataset = SpyFishAotearoaDataset(self.args.data_path, "train.csv", get_transform(train=True))
        dataset_test = SpyFishAotearoaDataset(self.args.data_path, "validation.csv", get_transform(train=False))

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.model.to(device)
        verbose = True  # todo: maybe change
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.args.learning_rate, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.learning_rate_size,
                                                       gamma=self.args.gamma)

        with wandb.init(config=vars(self.args)):
            for epoch in range(self.args.epochs):
                should_log = epoch % 10 == 0
                print('Epoch {} of {}'.format(epoch + 1, self.args.epochs))
                avg_train_loss = self.train_one_epoch(optimizer, data_loader, device, verbose)
                lr_scheduler.step()
                avg_val_loss, avg_iou = self.evaluate(data_loader_test, should_log, device)

                if verbose:
                    print(f'Losses of epoch num {epoch} are:')
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
            torch.save(self.model, self.args.output_path) # todo: Change path once training models

    def _eval_iou_and_log_img(self, images, targets):
        self.model.eval()
        results = self.model(images)

        img_boxes = []
        batch_iou_sum = 0
        n_boxes = 0

        for i, img in enumerate(images):
            img_boxes.append(add_bounding_boxes(img.cpu(), results[i]['labels'].cpu(), results[i]['scores'].cpu(), results[i]['boxes'].cpu()))
            batch_iou_sum += box_iou(targets[i]["boxes"].cpu(), results[i]["boxes"].cpu())
            n_boxes += results[i]["boxes"].shape[0]

        wandb.log({"classifications_images": [wandb.Image(image) for image in img_boxes]})

        return batch_iou_sum, n_boxes  # TODO: avg not right

    def train_one_epoch(self, optimizer, data_loader, device, verbose=True):
        self.model.train()

        iterable = tqdm(data_loader, position=0, leave=True) if verbose else data_loader
        avg_train_loss = 0

        for images, targets in iterable:
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

        return avg_train_loss / len(data_loader.dataset)

    def evaluate(self, val_set, img_log, device, verbose=True):
        # todo: check about to.device(CPU)
        # In order to get the validation loss we need to use .train()
        self.model.train()
        avg_iou = 0
        boxes_num = 0  # How many boxes the model found, for calculating IOU
        avg_val_loss = 0

        with torch.no_grad():
            if verbose:
                print('Starting validation')
            iterable = tqdm(val_set, position=0, leave=True) if verbose else val_set
            for images, targets in iterable:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                avg_val_loss += losses.item()

                if img_log:
                    iou_results = self._eval_iou_and_log_img(images, targets)
                    avg_iou += iou_results[0].sum()
                    boxes_num += iou_results[1]
                    self.model.train()

        return avg_val_loss, avg_iou / boxes_num if img_log else 0

    def predict(self, img_path, device, class_names, cls_tresh=0.5, iou_tresh=0.5):
        """
        Get prediction on images from dataloader
        :param dataloader
        :return: Classification and bounding boxes for each image provided
        """
        self.model.eval()
        with torch.no_grad():
            img = Image.open(img_path)
            transform = T.Compose([T.ToTensor()])
            img = transform(img).to(device)
            pred = self.model([img])
            pred_class = [class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
            pred_score = list(pred[0]['scores'].detach().cpu().numpy())

            pred_t = [pred_score.index(x) for x in pred_score if x > cls_tresh][-1]

            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]
            pred_score = pred_score[:pred_t + 1]

            images = add_bounding_boxes(img, pred_class, pred_score, pred_boxes, thresh=iou_tresh)
            return images, pred_boxes, pred_class, pred_score