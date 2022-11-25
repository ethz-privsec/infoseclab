import torch
from torchvision import transforms, models
from infoseclab.defenses import ResNet


class ResNetRandom(ResNet):
    """
    A defense that does a random pre-processing of the input before using a ResNet50 model.
    """
    def __init__(self, device):
        super().__init__(device)

        # extract a random crop of the image and resize it to (224, 224)
        # then add random noise to the image
        self.random_preproc = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)),
            lambda x: x + torch.randn_like(x) * 0.025 * 255.0
        ])

    def get_logits(self, x):
        return super().get_logits(self.random_preproc(x))
