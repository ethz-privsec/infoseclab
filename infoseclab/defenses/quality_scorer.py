from nima.inference.inference_model import InferenceModel
from torchvision import transforms
import torch


class ImageQualityScorer:
    def __init__(self):
        self.nima = InferenceModel(path_to_model="nima.pytorch/nima_model.pth")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def get_scores(self, x):
        scores = self.nima.model(self.normalize(x))
        buckets = torch.arange(1, 11).to(x.device)
        mu = (buckets * scores).sum(dim=1)
        return mu
