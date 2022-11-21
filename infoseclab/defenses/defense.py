import torch
from torchvision import transforms, models


class Defense(object):
    """
    An abstract class for a defense.
    """

    def __init__(self, device):
        self.device = device

    def get_logits(self, x):
        """
        Get the logits of the defense on a batch of images.
        :param x: the batch of images of size (batch_size, 3, 224, 224) in the range [0, 255]
        :return: the logits of the defense of size (batch_size, 1000)
        """
        raise NotImplementedError()

    def classify(self, x):
        """
        Get the classification of the defense on a batch of images.
        :param x: the batch of images of size (batch_size, 3, 224, 224) in the range [0, 255]
        :return: the classification of the defense of size (batch_size,)
        """
        return torch.argmax(self.get_logits(x), dim=-1)


class ResNet(Defense):
    """
    A defense that uses a ResNet50 model.
    """
    def __init__(self, device):
        super().__init__(device)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(device)
        # The classifier expects images normalized to 0 mean and 1 std.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def get_logits(self, x):
        return self.model(self.normalize(x / 255.0))
