import dataclasses
from enum import IntEnum
from typing import Callable

import cv2

from arcaea_offline_ocr.types import Mat


class ImageHashHashType(IntEnum):
    AVERAGE = 0
    DIFFERENCE = 1
    DCT = 2


class ImageHashCategory(IntEnum):
    JACKET = 0
    PARTNER_ICON = 1


@dataclasses.dataclass
class ImageHash:
    hash_type: ImageHashHashType
    category: ImageHashCategory
    label: str
    hash: bytes


@dataclasses.dataclass
class ImageHashResult:
    hash_type: ImageHashHashType
    category: ImageHashCategory
    label: str
    confidence: float


def _default_imread_gray(image_path: str):
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)


@dataclasses.dataclass
class ImageHashBuildTask:
    image_path: str
    category: ImageHashCategory
    label: str
    imread_function: Callable[[str], Mat] = _default_imread_gray
