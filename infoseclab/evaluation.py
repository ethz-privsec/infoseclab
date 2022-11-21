from infoseclab.data import ImageNet, EPSILON, npy_uint8_to_th
from infoseclab.defenses import ResNet, ResNetDetector, ResNetJPEG
from infoseclab.utils import batched_func
import torch
import numpy as np


def accuracy(defense, images, labels=ImageNet.labels):
    """
    Compute the accuracy of a defense on a set of images.
    :param defense: The defense to evaluate.
    :param images: The images to evaluate on, of shape (N, 3, 224, 224), in the range [0, 255].
    :param labels: The labels to evaluate on.
    :return: The accuracy of the defense on the images.
    """
    with torch.no_grad():
        all_preds = batched_func(defense.classify, images, defense.device, disable_tqdm=True)
        acc = torch.mean(torch.eq(labels, all_preds).float())
    return acc


def detector_accuracy(defense_det, clean_images, adv_images):
    """
    :param defense_det: the detector
    :param clean_images: the clean ImageNet images, of shape (N, 3, 224, 224), in the range [0, 255].
    :param adv_images: the adversarial images, of shape (N, 3, 224, 224), in the range [0, 255].
    :return: the accuracy of the detector on the clean and adversarial images
    """
    with torch.no_grad():
        clean_preds = batched_func(defense_det.detect, clean_images, defense_det.device, disable_tqdm=True)
        adv_preds = batched_func(defense_det.detect, adv_images, defense_det.device, disable_tqdm=True)
        acc_clean = torch.mean((clean_preds == 0).float())
        acc_adv = torch.mean((adv_preds == 0).float())

    print(f"\tclean detector accuracy: {100 * acc_clean}%")
    print(f"\tadv detector accuracy: {100 * acc_adv}%")

    return acc_clean, acc_adv


def assert_advs_valid(x_adv):
    """
    Assert that the adversarial images are valid.
    That is, the l_inf distance between the adversarial images and the clean
    images is less than or equal to our epsilon budget, and the images are
    in the range [0, 255].
    :param x_adv: the adversarial examples
    :return: True if the adversarial examples are valid
    """
    linf = torch.max(torch.abs(x_adv - ImageNet.clean_images))
    assert (torch.min(x_adv) >= 0.0) and (torch.max(x_adv) <= 255.0), "invalid pixel value"
    assert linf <= 1.01*EPSILON, "linf distance too large: {}".format(linf)
    return True


def load_and_validate_images(path):
    """
    Load and validate the adversarial images.
    :param path: the path to the adversarial images, saved as a uint8 numpy array
    :return: True if the adversarial images are valid
    """
    x_adv = np.load(path)
    x_adv = npy_uint8_to_th(x_adv)
    assert_advs_valid(x_adv)
    return x_adv


def eval_clf(clf, x_adv):
    acc_clean = accuracy(clf, ImageNet.clean_images)
    print(f"\tclean accuracy: {100 * acc_clean}%")

    acc_adv = accuracy(clf, x_adv)
    print(f"\tadv accuracy: {100 * acc_adv}%")

    acc_target = accuracy(clf, x_adv, ImageNet.targets)
    print(f"\tadv target accuracy: {100 * acc_target}%")

    return acc_clean, acc_adv, acc_target


def eval_untargeted_pgd(path="results/x_adv_untargeted.npy", device="cuda"):
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
    else:
        print("NOT THERE YET!")


def eval_targeted_pgd(path="results/x_adv_targeted.npy", device="cuda"):
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
    else:
        print("NOT THERE YET!")


def eval_detector(path="results/x_adv_detect.npy", detector_path="infoseclab/data/resnet18_detector.pth", device="cuda", ):
    print("=== Evaluating targeted PGD with Detection ===")
    defense_det = ResNetDetector(device, detector_path)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(defense_det, x_adv)

    assert acc_clean > 0.99, "clean accuracy too low"

    acc_det_clean, acc_det_adv = detector_accuracy(defense_det, ImageNet.clean_images, x_adv)
    assert acc_det_clean > 0.9, "clean detector accuracy too low"

    if (acc_adv < 0.01) and (acc_target > 0.99) and (acc_det_adv > 0.95):
        print("SUCCESS")
    else:
        print("NOT THERE YET!")


def eval_jpeg_attack(path="results/x_adv_jpeg.npy", device="cuda"):
    print("=== Evaluating targeted PGD on JPEG defense ===")
    defense_jpeg = ResNetJPEG(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target = eval_clf(defense_jpeg, x_adv)

    assert acc_clean > 0.9, "clean accuracy too low"

    if (acc_adv < 0.05) and (acc_target > 0.95):
        print("SUCCESS")
    else:
        print("NOT THERE YET!")


def main():
    device = "cuda"
    eval_untargeted_pgd(device=device)
    eval_targeted_pgd(device=device)
    eval_detector(device=device)
    eval_jpeg_attack(device=device)


if __name__ == "__main__":
    main()
