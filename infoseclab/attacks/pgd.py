import torch
from utils import batched_func


class PGD(object):
    """
    An untargeted PGD attack in l_inf norm.
    """
    def __init__(self, steps, epsilon, step_size, clf):
        """
        :param steps: the number of gradient steps to take
        :param epsilon: the maximum perturbation allowed
        :param step_size: the size of the gradient step
        :param clf: the classifier to attack
        """
        self.steps = steps
        self.epsilon = epsilon
        self.step_size = step_size
        self.clf = clf

    def project(self, x_adv, x_orig):
        """
        Project x_adv onto the epsilon ball around x_orig.
        :param x_adv: the adversarial images
        :param x_orig: the clean images
        :return: the adversarial images projected onto the epsilon ball
        """
        perturb = x_adv - x_orig
        perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
        return torch.clamp(x_orig + perturb, 0, 1)

    def attack_batch(self, x, y, verbose=False):
        """
        Attack a batch of images with untargeted PGD.
        :param x: the batch of images (torch tensors) to attack of size (batch_size, 3, 224, 224)
        :param y: the labels of the images to attack of size (batch_size,)
        :param verbose: whether to print the progress of the attack
        :return: the adversarial images of size (batch_size, 3, 224, 224)
        """

        # make a copy of the images that we will perturb
        x_adv = torch.clone(x)

        for i in range(self.steps):
            # we want gradients to flow all the way to the model inputs
            x_adv = x_adv.requires_grad_(True)

            # compute the current loss
            pred = self.clf.get_logits(x_adv)
            loss = torch.nn.CrossEntropyLoss()(pred, y)

            if verbose:
                confidence = torch.exp(-loss)
                print(f"{i}\t loss: {loss.item():.2e}, confidence: {confidence.item():.2e}")

            # compute the gradient of the loss with respect to the input pixels
            # and take an update step in the direction of the signed gradient
            grad, = torch.autograd.grad(loss, [x_adv])

            with torch.no_grad():
                # take a step in the direction of the gradient to maximize the loss
                x_adv = x_adv + self.step_size * torch.sign(grad)

                # project back onto the epsilon ball
                x_adv = self.project(x_adv, x)

        return x_adv.detach()

    def attack_all(self, images, labels):
        """
        Attack all images in the dataset.
        :param images: the images to attack
        :param labels: the clean labels
        :return:
        """
        return batched_func(self.attack_batch, inputs=(images, labels))

