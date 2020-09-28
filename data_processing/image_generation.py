import os

import cv2
import numpy as np
import torchvision
from torchvision import transforms
from tqdm import tqdm

from data_processing import pjoin


def generate_rotated_images(src: str) -> None:
    """
    Function generates new images with rotating existing ones within given path

    Args:
        :param src: path to directory where images are located.
            Generated images will be placed there with a specific prefixes.

    Returns:
        None
    """

    prefixes = ('_090', '_180', '_270')

    files = os.listdir(src)
    files = [f for f in files if f.endswith('.jpg')]

    for i, f in enumerate(files):
        img = cv2.imread(pjoin(src, f))
        for idx, angle in enumerate([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
            img = cv2.rotate(img, angle)
            filename, file_ext = os.path.splitext(f)
            img_name = pjoin(src, filename + prefixes[idx] + file_ext)
            cv2.imwrite(img_name, img)


def remove_bg(src: str) -> None:
    """
    Function loops over images within given path and deletes background on it

    Args:
        :param src: path to directory where images are located.

    Returns:
        None
    """

    files = os.listdir(src)
    files = [f for f in files if f.endswith('.jpg')]

    for f in tqdm(files):
        file_path = pjoin(src, f)
        img_original = cv2.imread(pjoin(src, f))
        img_cleaned = remove_image_bg(img_original)
        img_cleaned = np.array(img_cleaned)
        cv2.imwrite(file_path, img_cleaned)


def remove_image_bg(in_img):
    in_img = np.array(in_img)
    height, width = in_img.shape[:2]
    mask = np.zeros([height, width], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (15, 15, width - 30, height - 30)
    cv2.grabCut(in_img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    out_img = in_img * mask[:, :, np.newaxis]
    background = in_img - out_img
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    out_img = background + out_img

    return transforms.functional.to_pil_image(out_img)
