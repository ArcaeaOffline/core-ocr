from dataclasses import dataclass
from typing import Optional

from cv2 import COLOR_BGR2HSV, GaussianBlur, cvtColor, imread

from .crop import *
from .device import Device
from .mask import *
from .ocr import *


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
    pure_roi = mask_gray(pure_roi)
    pure_roi = GaussianBlur(pure_roi, (3, 3), 0)
    pure = ocr_pure(pure_roi)

    far_roi = crop_to_far(img_hsv, device)
    far_roi = mask_gray(far_roi)
    far_roi = GaussianBlur(far_roi, (3, 3), 0)
    far = ocr_far_lost(far_roi)

    lost_roi = crop_to_lost(img_hsv, device)
    lost_roi = mask_gray(lost_roi)
    lost_roi = GaussianBlur(lost_roi, (3, 3), 0)
    lost = ocr_far_lost(lost_roi)

    score_roi = crop_to_score(img_hsv, device)
    score_roi = mask_white(score_roi)
    score_roi = GaussianBlur(score_roi, (3, 3), 0)
    score = ocr_score(score_roi)

    max_recall_roi = crop_to_max_recall(img_hsv, device)
    max_recall_roi = mask_gray(max_recall_roi)
    max_recall = ocr_max_recall(max_recall_roi)

    rating_class_roi = crop_to_rating_class(img_hsv, device)
    rating_class_roi = mask_rating_class(rating_class_roi)
    rating_class = ocr_rating_class(rating_class_roi)

    title_roi = crop_to_title(img_hsv, device)
    title_roi = mask_white(title_roi)
    title = ocr_title(title_roi)

    return RecognizeResult(
        pure=pure,
        far=far,
        lost=lost,
        score=score,
        max_recall=max_recall,
        rating_class=rating_class,
        title=title,
    )
