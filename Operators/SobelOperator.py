import numpy as np
import cv2


vertical_mask = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
horizontal_mask = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])


def apply_vertical_mask(image):
    return cv2.filter2D(image, -1, vertical_mask).astype(np.uint8)


def apply_horizontal_mask(image):
    return cv2.filter2D(image, -1, horizontal_mask).astype(np.uint8)


def apply_both_masks(image):
    return apply_vertical_mask(image) + apply_horizontal_mask(image)


def calculate_sum(image):
    return np.sum(apply_both_masks(image))
