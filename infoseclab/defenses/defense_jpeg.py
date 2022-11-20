from io import BytesIO

from infoseclab.defenses import ResNetNima
from PIL import Image
import numpy as np
import torch


class ResNetNimaJPEG(ResNetNima):
    """
    A defense that converts images to JPEG format before classifying them or scoring them.
    """
    def __init__(self, device):
        super().__init__(device)

    def get_logits(self, x):
        return super().get_logits(images_to_JPEG(x))

    def get_scores(self, x):
        return super().get_scores(images_to_JPEG(x))


def _to_JPEG(im):
    """
    Convert a torch tensor image to JPEG format, in memory.
    """
    # convert torch tensor to PIL Image
    device = im.device
    im = Image.fromarray(np.uint8(im.cpu().numpy().transpose(1, 2, 0) * 255))

    # convert PIL Image to JPEG in memory
    with BytesIO() as f:
        im.save(f, format='JPEG')
        f.seek(0)
        im = Image.open(f)
        im.load()

    # convert PIL Image to torch tensor
    im = (np.asarray(im).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(im).to(device)


def images_to_JPEG(images):
    """
    Convert a batch of images to JPEG format.
    :param images: the images to convert as a torch tensor of dimension (N, 3, 224, 224)
    :return: the converted images as a torch tensor of dimension (N, 3, 224, 224)
    """
    with torch.no_grad():
        return torch.stack([_to_JPEG(im) for im in images])
