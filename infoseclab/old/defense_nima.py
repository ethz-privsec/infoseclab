from nima.inference.inference_model import InferenceModel
from torchvision import transforms
import torch

from infoseclab.defenses import ResNet


class ResNetNima(ResNet):
    """
    A defense that also assigns a quality score to an image.
    """
    def __init__(self, device):
        self.nima = InferenceModel(path_to_model="nima.pytorch/nima_model.pth")
        self.normalize_nima = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        super().__init__(device)

    def get_scores(self, x):
        scores = self.nima.model(self.normalize_nima(x))
        buckets = torch.arange(1, 11).to(x.device)
        mu = (buckets * scores).sum(dim=1)
        return mu


def image_quality(nima, images):
    """
    Compute the image quality of a set of images.
    :param nima: the NIMA model to use
    :param images: the images to evaluate of shape (N, 3, 224, 224), in the range [0, 255].
    :return: the image quality of the images
    """
    with torch.no_grad():
        return batched_func(nima.get_scores, images, nima.device, disable_tqdm=True)