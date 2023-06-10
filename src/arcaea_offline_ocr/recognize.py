from dataclasses import dataclass
from typing import Callable, Optional

from cv2 import COLOR_BGR2HSV, GaussianBlur, Mat, cvtColor, imread

from .crop import *
from .device import Device
from .mask import *
from .ocr import *


def process_digit_ocr_img(img_hsv, mask=Callable[[Mat], Mat]):
    img_hsv = mask(img_hsv)
    img_hsv = GaussianBlur(img_hsv, (3, 3), 0)
    return img_hsv


def process_tesseract_ocr_img(img_hsv, mask=Callable[[Mat], Mat]):
    img_hsv = mask(img_hsv)
    img_hsv = GaussianBlur(img_hsv, (1, 1), 0)
    return img_hsv


@dataclass(kw_only=True)
class RecognizeResult:
    pure: Optional[int]
    far: Optional[int]
    lost: Optional[int]
    score: Optional[int]
    max_recall: Optional[int]
    rating_class: Optional[int]
    title: str


def recognize(img_filename: str, device: Device):
    img = imread(img_filename)
    img_hsv = cvtColor(img, COLOR_BGR2HSV)

    pure_roi = crop_to_pure(img_hsv, device)
    pure = ocr_pure(process_digit_ocr_img(pure_roi, mask=mask_gray))

    far_roi = crop_to_far(img_hsv, device)
    far = ocr_far_lost(process_digit_ocr_img(far_roi, mask=mask_gray))

    lost_roi = crop_to_lost(img_hsv, device)
    lost = ocr_far_lost(process_digit_ocr_img(lost_roi, mask=mask_gray))

    score_roi = crop_to_score(img_hsv, device)
    score = ocr_score(process_digit_ocr_img(score_roi, mask=mask_white))

    max_recall_roi = crop_to_max_recall(img_hsv, device)
    max_recall = ocr_max_recall(
        process_tesseract_ocr_img(max_recall_roi, mask=mask_gray)
    )

    rating_class_roi = crop_to_rating_class(img_hsv, device)
    rating_class = ocr_rating_class(
        process_tesseract_ocr_img(rating_class_roi, mask=mask_rating_class)
    )

    title_roi = crop_to_title(img_hsv, device)
    title = ocr_title(process_tesseract_ocr_img(title_roi, mask=mask_white))

    return RecognizeResult(
        pure=pure,
        far=far,
        lost=lost,
        score=score,
        max_recall=max_recall,
        rating_class=rating_class,
        title=title,
    )
