import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from general_utils import get_transform_style
from style_transfer_lib import Normalization, ContentLoss, StyleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# desired size of the output image
IM_SIZE = (960, 1280)
# desired depth layers to compute style/content losses :
CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def transfer_single_dataset(data_path, style_path, output_images_path, style_name):
    """
    A function that transfer a single dataset with specified style
    """
    loader = get_transform_style(IM_SIZE)
    unloader = T.ToPILImage()  # reconvert into PIL image

    content_images_names = sorted(os.listdir(data_path))

    style_image = load_style_images(style_path, loader)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Do style transfer for all the content images and save in drive
    for i, content_img in enumerate(content_images_names):
        print(f"transferring {i}/{len(content_images_names)} to style {style_name}")
        torch.cuda.empty_cache()
        input_img = image_loader(os.path.join(data_path, content_img), loader)
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    input_img, T.Resize(input_img.shape[-2:])(style_image), input_img)
        image = output.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image_name = style_name + "_"+ content_images_names[i]
        output_path = output_images_path.rsplit("/", 1)
        output_path = output_path[0]
        image.save(os.path.join(output_path,"style_images",image_name))


def transfer_datasets(data_path, style_images_path, input_csv, output_images_path, output_csv_path):
    """
    Transfer several dataset into the style specified
    """
    for style_image in os.listdir(style_images_path):
        transfer_single_dataset(os.path.join(data_path, "images"), os.path.join(style_images_path, style_image),
                                output_images_path, style_image.rsplit(".")[0])

    # create csv file
    original_table = pd.read_csv(input_csv)
    rows_num = original_table.shape[0]

    styles_reps = len(os.listdir(style_images_path)) + 1 # adding one because we want the original images to stay in the csv

    style_table = pd.concat([original_table] * styles_reps, ignore_index=True)

    low, high = rows_num, rows_num * 2 - 1

    for style_image in os.listdir(style_images_path):
        style_table.loc[low:high, "image_name"] = style_image + style_table.loc[
                                                              low:high,
                                                              "image_name"]
        low += rows_num
        high += rows_num

    style_table.to_csv(output_csv_path, sep=",")

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=None,
                               style_layers=None):
    """
    Function creating the model and two kinds of losses- style and content
    Returns:
    """
    # normalization module
    if style_layers is None:
        style_layers = STYLE_LAYERS_DEFAULT
    if content_layers is None:
        content_layers = CONTENT_LAYERS_DEFAULT

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=250,
                       style_weight=10000, content_weight=20):
    """
    Run the style transfer
    Returns: The image with style

    """
    print('Building the style transfer model')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std,
                                                                     style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def get_input_optimizer(input_img):
    """
    A function to create the optimizer of the specifed image
    Returns: the optimizer image
    """
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def image_loader(image_name, loader):
    """
    Function to load the specified image
    Returns: The tensor image connected to the device
    """
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def create_content(content_path, loader):
    """
    Function to create the content images and a list of their names
    Returns: A list of the images and a list for the name of the images

    """
    content_images = []
    content_img_names = os.listdir(content_path)
    for img_path in content_img_names:
        content_images.append(image_loader(os.path.join(content_path, img_path), loader))

    return content_images, content_img_names


def load_style_images(style_img_path, loader):
    """
    Loads the style image
    Returns: The style image loaded

    """
    style_img = image_loader(style_img_path, loader)
    style_img = style_img[:, :3, :, :]  # check why
    return style_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--data_path",
        type=str,
        default="./data_0808",
        help="The path of the data directory")

    parser.add_argument(
        "-s",
        "--style_images_path",
        type=str,
        help="The path of the style images",
        default="./data/style_images"
    )

    parser.add_argument(
        "-ic",
        "--input_csv",
        type=str,
        help="Path to csv file",
        default="./data_0808/output/train.csv"
    )

    parser.add_argument(
        "-oi",
        "--output_images_path",
        type=str,
        help="Save the generated images to",
        default="./data_0808/images/style_images"
    )

    parser.add_argument(
        "-oc",
        "--output_csv_path",
        type=str,
        help="The output path of the '.csv' files",
        default="./data_0808/output/style_train.csv"
    )

    args = parser.parse_args()

    # generating the default output folder if not exists
    if not os.path.isdir(args.output_images_path):
        os.mkdir(args.output_images_path)

    transfer_datasets(args.data_path, args.style_images_path, args.input_csv, args.output_images_path,
                      args.output_csv_path)

if __name__ == '__main__':
    main()
