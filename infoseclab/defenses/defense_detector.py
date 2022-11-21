import torch
import torch.nn as nn
import copy
from infoseclab.defenses import ResNet


class ResNetDetector(ResNet):
    """
    A defense that also aims to detect adversarial examples.
    """
    def __init__(self, device, detector_path):
        """
        :param device: the device to use for the defense
        :param detector_path: the path to the detector model
        """
        super().__init__(device)

        # save the classifier head of the model
        self.fc = copy.deepcopy(self.model.fc)

        # remove the classifier head so that we get the penultimate layer features
        self.model.fc = torch.nn.Identity()

        # load the detector model
        det = nn.Linear(2048, 2)
        det.load_state_dict(torch.load(detector_path, map_location=device))
        self.det = det.to(device)

    def get_logits(self, x):
        # get the penultimate layer features and pass them through the classifier head
        return self.fc(super().get_logits(x))

    def get_detection_logits(self, x):
        # get the penultimate layer features and pass them through the detector
        return self.det(super().get_logits(x))

    def detect(self, x):
        """
        Detect adversarial examples.
        :param x: the batch of images of size (batch_size, 3, 224, 224) in the range [0, 255]
        :return: the detection of the defense of size (batch_size,). 0 means clean, 1 means adversarial.
        """
        return torch.argmax(self.get_detection_logits(x), dim=-1)
