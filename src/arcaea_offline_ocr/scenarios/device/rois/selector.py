from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np
from typing_extensions import override

from arcaea_offline_ocr.scenarios.device.masker.auto import (
    DeviceRoisMaskerAutoT1,
    DeviceRoisMaskerAutoT2,
)

from .auto import DeviceRoisAutoT1, DeviceRoisAutoT2

if TYPE_CHECKING:
    from collections.abc import Callable

    from cv2.typing import MatLike

    from arcaea_offline_ocr.scenarios.device.masker.base import DeviceRoisMasker

    from .base import DeviceRois


class DeviceRoisAutoSelectorResult(Enum):
    T1 = "T1"
    T2 = "T2"
    UNKNOWN = "UNKNOWN"


def crop_rounded(img: MatLike, rect: tuple[float, float, float, float]) -> MatLike:
    x, y, w, h = (round(v) for v in rect)
    return img[y : y + h, x : x + w]


def pfl_label_rect(roi: tuple[float, float, float, float], label_width: float):
    x, y, _, h = roi
    return (x - label_width, y, label_width, h)


class DeviceRoisAutoSelectorDetector(ABC):
    result: ClassVar[DeviceRoisAutoSelectorResult]

    @abstractmethod
    def confidence(self, img_bgr: MatLike) -> float:
        """
        Analyze the img_bgr, then return the confidence of the img_bgr.

        For example, if the detector has 3 interest points:

        - PURE label should be blue,
        - FAR label should be gray,
        - LOST label should be gray

        and if 2 of them matches, then the return value would be ``0.66``.
        """


class DeviceRoisAutoSelectorDetectorT1(DeviceRoisAutoSelectorDetector):
    INTEREST_POINTS: ClassVar[float] = 3.0

    PFL_LABEL_WIDTH: ClassVar[int] = 85
    PURE_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([80, 60, 125], np.uint8)
    PURE_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([110, 200, 225], np.uint8)

    PURE_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.08
    FAR_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.04
    LOST_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.055

    result: ClassVar[DeviceRoisAutoSelectorResult] = DeviceRoisAutoSelectorResult.T1

    _masker: ClassVar[DeviceRoisMaskerAutoT1] = DeviceRoisMaskerAutoT1()

    def _mask_pure_label(self, label_bgr: MatLike):
        label_hsv = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(
            label_hsv,
            self.PURE_LABEL_HSV_LOWER,
            self.PURE_LABEL_HSV_UPPER,
        )

    def _mask_far_lost_label(self, label_bgr: MatLike):
        return self._masker.gray(label_bgr)

    @override
    def confidence(self, img_bgr: MatLike) -> float:
        rois = DeviceRoisAutoT1(img_bgr.shape[1], img_bgr.shape[0])
        label_width = self.PFL_LABEL_WIDTH * rois.factor

        interests: list[
            tuple[
                tuple[float, float, float, float], Callable[[MatLike], MatLike], float
            ]
        ] = [
            (
                rois.pure,
                self._mask_pure_label,
                self.PURE_LABEL_NON_ZERO_RATIO_THRESHOLD,
            ),
            (
                rois.far,
                self._mask_far_lost_label,
                self.FAR_LABEL_NON_ZERO_RATIO_THRESHOLD,
            ),
            (
                rois.lost,
                self._mask_far_lost_label,
                self.LOST_LABEL_NON_ZERO_RATIO_THRESHOLD,
            ),
        ]

        matches = 0
        for roi, masker, threshold in interests:
            label_bgr = crop_rounded(img_bgr, pfl_label_rect(roi, label_width))
            label_masked = masker(label_bgr)
            if np.count_nonzero(label_masked) / label_masked.size >= threshold:
                matches += 1

        return matches / self.INTEREST_POINTS


class DeviceRoisAutoSelectorDetectorT2(DeviceRoisAutoSelectorDetector):
    INTEREST_POINTS: ClassVar[float] = 3.0

    PFL_LABEL_WIDTH: ClassVar[int] = 180
    PURE_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([110, 25, 90], np.uint8)
    PURE_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([160, 150, 230], np.uint8)
    FAR_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([5, 25, 120], np.uint8)
    FAR_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([20, 100, 240], np.uint8)
    LOST_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([160, 5, 190], np.uint8)
    LOST_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([179, 60, 255], np.uint8)

    PFL_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.3

    result: ClassVar[DeviceRoisAutoSelectorResult] = DeviceRoisAutoSelectorResult.T2

    @override
    def confidence(self, img_bgr: MatLike) -> float:
        rois = DeviceRoisAutoT2(img_bgr.shape[1], img_bgr.shape[0])
        label_width = self.PFL_LABEL_WIDTH * rois.factor

        interests: list[tuple[tuple[float, float, float, float], MatLike, MatLike]] = [
            (rois.pure, self.PURE_LABEL_HSV_LOWER, self.PURE_LABEL_HSV_UPPER),
            (rois.far, self.FAR_LABEL_HSV_LOWER, self.FAR_LABEL_HSV_UPPER),
            (rois.lost, self.LOST_LABEL_HSV_LOWER, self.LOST_LABEL_HSV_UPPER),
        ]

        matches = 0
        for roi, hsv_lower, hsv_upper in interests:
            label_bgr = crop_rounded(img_bgr, pfl_label_rect(roi, label_width))
            label_hsv = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2HSV)
            label_masked = cv2.inRange(label_hsv, hsv_lower, hsv_upper)
            if np.count_nonzero(label_masked) / label_masked.size >= (
                self.PFL_LABEL_NON_ZERO_RATIO_THRESHOLD
            ):
                matches += 1

        return matches / self.INTEREST_POINTS


class DeviceRoisAutoSelector:
    selectors: ClassVar[list[DeviceRoisAutoSelectorDetector]] = [
        DeviceRoisAutoSelectorDetectorT1(),
        DeviceRoisAutoSelectorDetectorT2(),
    ]

    @classmethod
    def select(
        cls,
        img_bgr: MatLike,
        fallback: DeviceRoisAutoSelectorResult | None = None,
    ) -> DeviceRoisAutoSelectorResult:
        results = {selector: selector.confidence(img_bgr) for selector in cls.selectors}

        if all(confidence == 0.0 for confidence in results.values()):
            return fallback or DeviceRoisAutoSelectorResult.UNKNOWN
        return max(results.items(), key=lambda it: it[1])[0].result

    @staticmethod
    def create_rois_and_masker(
        result: DeviceRoisAutoSelectorResult,
        w: int,
        h: int,
    ) -> tuple[DeviceRois, DeviceRoisMasker]:
        """Create the rois and masker pair matching the selector result."""
        if result is DeviceRoisAutoSelectorResult.T1:
            return DeviceRoisAutoT1(w, h), DeviceRoisMaskerAutoT1()
        if result is DeviceRoisAutoSelectorResult.T2:
            return DeviceRoisAutoT2(w, h), DeviceRoisMaskerAutoT2()
        msg = f"no rois/masker pair for {result}"
        raise ValueError(msg)
