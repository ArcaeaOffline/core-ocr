import cv2
from numpy import array, max, min, uint8

from .types import Mat

__all__ = [
    "GRAY_MIN_HSV",
    "GRAY_MAX_HSV",
    "WHITE_MIN_HSV",
    "WHITE_MAX_HSV",
    "PST_MIN_HSV",
    "PST_MAX_HSV",
    "PRS_MIN_HSV",
    "PRS_MAX_HSV",
    "FTR_MIN_HSV",
    "FTR_MAX_HSV",
    "BYD_MIN_HSV",
    "BYD_MAX_HSV",
    "mask_gray",
    "mask_white",
    "mask_pst",
    "mask_prs",
    "mask_ftr",
    "mask_byd",
    "mask_rating_class",
]

GRAY_MIN_HSV = array([0, 0, 70], uint8)
GRAY_MAX_HSV = array([0, 0, 200], uint8)

GRAY_MIN_BGR = array([50] * 3, uint8)
GRAY_MAX_BGR = array([160] * 3, uint8)

WHITE_MIN_HSV = array([0, 0, 240], uint8)
WHITE_MAX_HSV = array([179, 10, 255], uint8)

PST_MIN_HSV = array([100, 50, 80], uint8)
PST_MAX_HSV = array([100, 255, 255], uint8)

PRS_MIN_HSV = array([43, 40, 75], uint8)
PRS_MAX_HSV = array([50, 155, 190], uint8)

FTR_MIN_HSV = array([149, 30, 0], uint8)
FTR_MAX_HSV = array([155, 181, 150], uint8)

BYD_MIN_HSV = array([170, 50, 50], uint8)
BYD_MAX_HSV = array([179, 210, 198], uint8)


def mask_gray_new(__img_bgr: Mat):
    # bgr_value_equal_mask = all(__img_bgr[:, 1:] == __img_bgr[:, :-1], axis=1)
    bgr_value_equal_mask = max(__img_bgr, axis=2) - min(__img_bgr, axis=2) <= 5
    img_bgr = __img_bgr.copy()
    img_bgr[~bgr_value_equal_mask] = array([0, 0, 0], __img_bgr.dtype)
    img_bgr = cv2.erode(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_bgr = cv2.dilate(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    return cv2.inRange(img_bgr, GRAY_MIN_BGR, GRAY_MAX_BGR)


def mask_gray(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, GRAY_MIN_HSV, GRAY_MAX_HSV)
    mask = cv2.dilate(mask, (2, 2))
    return mask


def mask_white(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, WHITE_MIN_HSV, WHITE_MAX_HSV)
    mask = cv2.dilate(
        mask, (5, 5), borderType=cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED
    )
    return mask


def mask_pst(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, PST_MIN_HSV, PST_MAX_HSV)
    mask = cv2.dilate(mask, (1, 1))
    return mask


def mask_prs(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, PRS_MIN_HSV, PRS_MAX_HSV)
    mask = cv2.dilate(mask, (1, 1))
    return mask


def mask_ftr(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, FTR_MIN_HSV, FTR_MAX_HSV)
    mask = cv2.dilate(mask, (1, 1))
    return mask


def mask_byd(img_hsv: Mat):
    mask = cv2.inRange(img_hsv, BYD_MIN_HSV, BYD_MAX_HSV)
    mask = cv2.dilate(mask, (2, 2))
    return mask


def mask_rating_class(img_hsv: Mat):
    pst = mask_pst(img_hsv)
    prs = mask_prs(img_hsv)
    ftr = mask_ftr(img_hsv)
    byd = mask_byd(img_hsv)
    return cv2.bitwise_or(byd, cv2.bitwise_or(ftr, cv2.bitwise_or(pst, prs)))
