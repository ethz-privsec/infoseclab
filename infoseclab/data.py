import numpy as np
import torch
import json

EPSILON = 8./255.  # Our perturbation budget is 8/255.0 for each pixel.


def th_to_npy_uint8(x):
    """
    Convert a torch tensor to a numpy array of uint8 values.
    :param x: a torch tensor of floats in the range [0, 1]
    :return: a numpy array of uint8 values
    """
    return np.rint(x.detach().cpu().numpy() * 255.0).astype(np.uint8)


def npy_uint8_to_th(x):
    """
    Convert a numpy array of uint8 values to a torch tensor.
    :param x: a numpy array of uint8 values
    :return: a torch tensor of floats in the range [0, 1]
    """
    assert x.dtype == np.uint8
    return torch.from_numpy(x.astype(np.float32) / 255.0)


class ImageNet:
    """
    A subset of 200 images from the ImageNet validation set.
    """

    # get the labels corresponding to the 1000 ImageNet classes
    with open("data/imagenet-simple-labels.json") as f:
        class_names = json.load(f)

    # load the images and labels
    clean_images = npy_uint8_to_th(np.load("data/images.npy"))
    labels = torch.from_numpy(np.load("data/labels.npy"))

    # attack targets
    targets = torch.from_numpy(np.load("data/targets.npy"))

