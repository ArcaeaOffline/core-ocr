from dataclasses import dataclass
from typing import Callable, Optional

from cv2 import COLOR_BGR2HSV, GaussianBlur, Mat, cvtColor, imread

from .crop import *
from .device import Device
from .mask import *
from .ocr import *
from .utils import imread_unicode

__all__ = [
    "process_digits_ocr_img",
    "process_tesseract_ocr_img",
    "recognize_pure",
    "recognize_far_lost",
    "recognize_score",
    "recognize_max_recall",
    "recognize_rating_class",
    "recognize_title",
    "RecognizeResult",
    "recognize",
]


def process_digits_ocr_img(img_hsv_cropped: Mat, mask=Callable[[Mat], Mat]):
    img_hsv_cropped = mask(img_hsv_cropped)
    img_hsv_cropped = GaussianBlur(img_hsv_cropped, (3, 3), 0)
    return img_hsv_cropped


def process_tesseract_ocr_img(img_hsv_cropped: Mat, mask=Callable[[Mat], Mat]):
    img_hsv_cropped = mask(img_hsv_cropped)
    img_hsv_cropped = GaussianBlur(img_hsv_cropped, (1, 1), 0)
    return img_hsv_cropped


def recognize_pure(img_hsv_cropped: Mat):
    return ocr_pure(process_digits_ocr_img(img_hsv_cropped, mask=mask_gray))


def recognize_far_lost(img_hsv_cropped: Mat):
    return ocr_far_lost(process_digits_ocr_img(img_hsv_cropped, mask=mask_gray))


def recognize_score(img_hsv_cropped: Mat):
    return ocr_score(process_digits_ocr_img(img_hsv_cropped, mask=mask_white))


def recognize_max_recall(img_hsv_cropped: Mat):
    return ocr_max_recall(process_tesseract_ocr_img(img_hsv_cropped, mask=mask_gray))


def recognize_rating_class(img_hsv_cropped: Mat):
    return ocr_rating_class(
        process_tesseract_ocr_img(img_hsv_cropped, mask=mask_rating_class)
    )


def recognize_title(img_hsv_cropped: Mat):
    return ocr_title(process_tesseract_ocr_img(img_hsv_cropped, mask=mask_white))


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
    img = imread_unicode(img_filename)
    img_hsv = cvtColor(img, COLOR_BGR2HSV)

    pure_roi = crop_to_pure(img_hsv, device)
    pure = recognize_pure(pure_roi)

    far_roi = crop_to_far(img_hsv, device)
    far = recognize_far_lost(far_roi)

    lost_roi = crop_to_lost(img_hsv, device)
    lost = recognize_far_lost(lost_roi)

    score_roi = crop_to_score(img_hsv, device)
    score = recognize_score(score_roi)

    max_recall_roi = crop_to_max_recall(img_hsv, device)
    max_recall = recognize_max_recall(max_recall_roi)

    rating_class_roi = crop_to_rating_class(img_hsv, device)
    rating_class = recognize_rating_class(rating_class_roi)

    title_roi = crop_to_title(img_hsv, device)
    title = recognize_title(title_roi)

    return RecognizeResult(
        pure=pure,
        far=far,
        lost=lost,
        score=score,
        max_recall=max_recall,
        rating_class=rating_class,
        title=title,
    )
