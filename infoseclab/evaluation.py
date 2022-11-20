from infoseclab.data import  ImageNet, EPSILON, npy_uint8_to_th
from infoseclab.defenses.defense import ResNet
from infoseclab.defenses.defense_jpeg import ResNetJPEG
from infoseclab.defenses.quality_scorer import ImageQualityScorer
from infoseclab.utils import batched_func
import torch
import numpy as np

device = "cuda"


def accuracy(defense, images, labels=ImageNet.labels):
    """
    Compute the accuracy of a defense on a set of images.
    :param defense: The defense to evaluate.
    :param images: The images to evaluate on, of shape (N, 3, 224, 224).
    :param labels: The labels to evaluate on.
    :return: The accuracy of the defense on the images.
    """
    with torch.no_grad():
        all_preds = batched_func(defense.classify, images, verbose=False)
        acc = torch.mean(torch.eq(labels, all_preds).float())
    return acc


def image_quality(nima, images):
    """
    Compute the image quality of a set of images.
    :param nima: the NIMA model to use
    :param images: the images to evaluate of shape (N, 3, 224, 224)
    :return: the image quality of the images
    """
    with torch.no_grad():
        return batched_func(nima.get_scores, images, verbose=False)


def assert_advs_valid(x_adv):
    """
    Assert that the adversarial images are valid.
    That is, the l_inf distance between the adversarial images and the clean
    images is less than or equal to our epsilon budget, and the images are
    in the range [0, 1].
    :param x_adv: the adversarial examples
    :return: True if the adversarial examples are valid
    """
    linf = torch.max(torch.abs(x_adv - ImageNet.clean_images))
    assert (torch.min(x_adv) >= 0.0) and (torch.max(x_adv) <= 1.0), "invalid pixel value"
    assert linf <= 1.01*EPSILON, "linf distance too large: {}".format(linf)
    return True


def load_and_validate_images(path):
    x_adv = np.load(path)
    x_adv = npy_uint8_to_th(x_adv)
    assert_advs_valid(x_adv)
    return x_adv


def eval_clf(clf, x_adv):
    acc_clean = accuracy(clf, ImageNet.clean_images)
    print(f"\nclean accuracy: {100 * acc_clean}%")

    acc_adv = accuracy(clf, x_adv)
    print(f"\nadv accuracy: {100 * acc_adv}%")

    acc_target = accuracy(clf, x_adv, ImageNet.targets)
    print(f"\nadv target accuracy: {100 * acc_target}%")

    return acc_clean, acc_adv, acc_target

def eval_untargeted_pgd(path="results/x_adv_untargeted.npy"):
    print("=== Evaluating untargeted PGD ===")
    resnet = ResNet(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(resnet, x_adv)

    assert acc_clean > 0.99, "clean accuracy too low"

    if acc_adv < 0.01:
        print("SUCCESS")
    print("NOT THERE YET!")


def eval_targeted_pgd(path="results/x_adv_targeted.npy"):
    print("=== Evaluating targeted PGD ===")
    resnet = ResNet(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(resnet, x_adv)

    assert acc_clean > 0.99, "clean accuracy too low"

    if (acc_adv < 0.01) and (acc_target > 0.99):
        print("SUCCESS")
    print("NOT THERE YET!")


def eval_nima_attack(path="results/x_adv_nima.npy"):
    print("=== Evaluating targeted PGD with Image Quality Assessment ===")
    nima = ImageQualityScorer()
    resnet = ResNet(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(resnet, x_adv)

    assert acc_clean > 0.99, "clean accuracy too low"

    scores_adv = image_quality(nima, x_adv)
    print(f"\nadv image quality: {scores_adv.mean()}, {scores_adv.min()}%")

    if (acc_adv < 0.01) and (acc_target > 0.99) and (scores_adv.min() > 6.9):
        print("SUCCESS")
    print("NOT THERE YET!")


def eval_jpeg_attack(path="results/x_adv_jpeg.npy"):
    print("=== Evaluating targeted PGD on JPEG defense ===")
    resnet_jpeg = ResNetJPEG(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(resnet_jpeg, x_adv)

    assert acc_clean > 0.9, "clean accuracy too low"

    if (acc_adv < 0.05) and (acc_target > 0.95):
        print("SUCCESS")
    print("NOT THERE YET!")


def eval_random_preproc_attack():
    pass


def main():
    eval_untargeted_pgd()
    eval_targeted_pgd()
    eval_nima_attack()
    eval_jpeg_attack()
    eval_random_preproc_attack()


if __name__ == "__main__":
    main()
