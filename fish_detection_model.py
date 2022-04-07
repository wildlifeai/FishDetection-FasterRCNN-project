import os
import wandb
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from SpyFishAotearoaDataset import SpyFishAotearoaDataset
import utils.transformers as T
from utils.general_utils import collate_fn
from torchvision.ops import box_iou


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

        # Creating data loader
        dataset = SpyFishAotearoaDataset(self.args.root_path, get_transform(train=True))
        dataset_test = SpyFishAotearoaDataset(self.args.root_path, get_transform(train=False))

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
                print('Epoch {} of {}'.format(epoch + 1, self.args.epochs))
                avg_train_loss = self.train_one_epoch(optimizer, data_loader, device, verbose)
                lr_scheduler.step()
                avg_iou = self.evaluate(data_loader_test, device)

                if verbose:
                    print(f'Losses of epoch num {epoch} are:')
                    print('Train loss: {}'.format(avg_train_loss))
                    print('Validation average IOU: {}'.format(avg_iou))

                wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "validation_avg_iou": avg_iou})

            # Saving the model in the specified path
            torch.save(self.model, self.args.output_path)

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

    def evaluate(self, val_set, device, verbose=True):
        # todo: check about to.device(CPU)
        self.model.eval()

        avg_iou = 0
        boxes_num = 0  # How many boxes the model found, for calculating IOU
        with torch.no_grad():
            if verbose:
                print('Starting validation')
            iterable = tqdm(val_set, position=0, leave=True) if verbose else val_set
            for images, targets in iterable:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                response = self.model(images)
                batch_iou = box_iou(targets[0]["boxes"], response[0]["boxes"])
                avg_iou += torch.sum(batch_iou).item()
                boxes_num += response[0]["boxes"].shape[1]

        return avg_iou / boxes_num

    def predict(self):
        pass
