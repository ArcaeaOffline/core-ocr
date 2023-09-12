import io
from collections.abc import Iterable
from typing import Callable, Tuple, TypeVar, Union, overload

import cv2
import numpy as np
from PIL import Image, ImageCms

from .types import Mat, XYWHRect

__all__ = ["imread_unicode"]


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_UNCHANGED) -> Mat:
    # https://stackoverflow.com/a/57872297/16484891
    # CC BY-SA 4.0
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)


def construct_int_xywh_rect(
    rect: XYWHRect, func: Callable[[Union[int, float]], int] = round
):
    return XYWHRect(*[func(num) for num in rect])


@overload
def apply_factor(item: int, factor: float) -> float:
    ...


@overload
def apply_factor(item: float, factor: float) -> float:
    ...


T = TypeVar("T", bound=Iterable)


@overload
def apply_factor(item: T, factor: float) -> T:
    ...


def apply_factor(item, factor: float):
    if isinstance(item, (int, float)):
        return item * factor
    elif isinstance(item, Iterable):
        return item.__class__([i * factor for i in item])


def convert_to_srgb(pil_img: Image.Image):
    """
    Convert PIL image to sRGB color space (if possible)
    and save the converted file.

    https://stackoverflow.com/a/65667797/16484891

    CC BY-SA 4.0
    """
    icc = pil_img.info.get("icc_profile", "")
    icc_conv = ""

    if icc:
        io_handle = io.BytesIO(icc)  # virtual file
        src_profile = ImageCms.ImageCmsProfile(io_handle)
        dst_profile = ImageCms.createProfile("sRGB")
        img_conv = ImageCms.profileToProfile(pil_img, src_profile, dst_profile)
        icc_conv = img_conv.info.get("icc_profile", "")

    return img_conv if icc != icc_conv else pil_img
