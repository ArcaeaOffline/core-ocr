import cv2

from arcaea_offline_ocr.types import Mat


def _resize_image(src: Mat, dsize: ...) -> Mat:
    return cv2.resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_AREA)
