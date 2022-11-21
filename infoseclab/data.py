import numpy as np
import torch
import json

EPSILON = 8.  # Our perturbation budget is 8 (out of 256) for each pixel.


def th_to_npy_uint8(x):
    """
    Convert a torch tensor to a numpy array of uint8 values.
    :param x: a torch tensor of floats in the range [0, 255]
    :return: a numpy array of uint8 values
    """
    return np.rint(x.detach().cpu().numpy()).astype(np.uint8)


def npy_uint8_to_th(x):
    """
    Convert a numpy array of uint8 values to a torch tensor.
    :param x: a numpy array of uint8 values
    :return: a torch tensor of floats in the range [0, 255]
    """
    assert x.dtype == np.uint8
    return torch.from_numpy(x.astype(np.float32))


class ImageNet:
    """
    A subset of 200 images from the ImageNet validation set.
    """

    # get the labels corresponding to the 1000 ImageNet classes
    with open("infoseclab/data/imagenet-simple-labels.json") as f:
        class_names = json.load(f)

    # load the images and labels
    clean_images = npy_uint8_to_th(np.load("infoseclab/data/images.npy"))
    labels = torch.from_numpy(np.load("infoseclab/data/labels.npy"))

    # attack targets
    targets = torch.from_numpy(np.load("infoseclab/data/targets.npy"))
