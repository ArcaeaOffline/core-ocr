from cv2 import BORDER_CONSTANT, BORDER_ISOLATED, Mat, bitwise_or, dilate, inRange
from numpy import array, uint8

GRAY_MIN_HSV = array([0, 0, 70], uint8)
GRAY_MAX_HSV = array([0, 70, 200], uint8)

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


def mask_gray(img_hsv: Mat):
    mask = inRange(img_hsv, GRAY_MIN_HSV, GRAY_MAX_HSV)
    mask = dilate(mask, (2, 2))
    return mask


def mask_white(img_hsv: Mat):
    mask = inRange(img_hsv, WHITE_MIN_HSV, WHITE_MAX_HSV)
    mask = dilate(mask, (5, 5), borderType=BORDER_CONSTANT | BORDER_ISOLATED)
    return mask


def mask_pst(img_hsv: Mat):
    mask = inRange(img_hsv, PST_MIN_HSV, PST_MAX_HSV)
    mask = dilate(mask, (1, 1))
    return mask


def mask_prs(img_hsv: Mat):
    mask = inRange(img_hsv, PRS_MIN_HSV, PRS_MAX_HSV)
    mask = dilate(mask, (1, 1))
    return mask


def mask_ftr(img_hsv: Mat):
    mask = inRange(img_hsv, FTR_MIN_HSV, FTR_MAX_HSV)
    mask = dilate(mask, (1, 1))
    return mask


def mask_byd(img_hsv: Mat):
    mask = inRange(img_hsv, BYD_MIN_HSV, BYD_MAX_HSV)
    mask = dilate(mask, (2, 2))
    return mask


def mask_rating_class(img_hsv: Mat):
    pst = mask_pst(img_hsv)
    prs = mask_prs(img_hsv)
    ftr = mask_ftr(img_hsv)
    byd = mask_byd(img_hsv)
    return bitwise_or(byd, bitwise_or(ftr, bitwise_or(pst, prs)))
