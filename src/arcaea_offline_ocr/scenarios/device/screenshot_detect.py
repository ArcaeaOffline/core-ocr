from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from .rois.auto import DeviceRoisAutoT2

if TYPE_CHECKING:
    from cv2.typing import MatLike


class ScreenshotDetect:
    PURPLE_HSV_LOWER: ClassVar[np.ndarray] = np.array([110, 45, 75], np.uint8)
    PURPLE_HSV_UPPER: ClassVar[np.ndarray] = np.array([140, 150, 175], np.uint8)

    ROI_HEIGHT_OFFSET: ClassVar[int] = -510
    ROI_WIDTH_OFFSET: ClassVar[int] = -250
    ROI_HEIGHT: ClassVar[int] = 165

    @classmethod
    def is_arcaea_screenshot(
        cls,
        img_hsv: MatLike,
        ratio: float = 0.65,
    ) -> bool:
        rois = DeviceRoisAutoT2(img_hsv.shape[1], img_hsv.shape[0])

        y = rois.layout_area_h_mid + cls.ROI_HEIGHT_OFFSET * rois.factor
        w = rois.w_mid + cls.ROI_WIDTH_OFFSET * rois.factor
        h = cls.ROI_HEIGHT * rois.factor

        roi = img_hsv[
            round(y) : round(y) + round(h),
            0 : round(w),
        ]
        mask = cv2.inRange(roi, cls.PURPLE_HSV_LOWER, cls.PURPLE_HSV_UPPER)

        return bool(np.count_nonzero(mask) >= mask.size * ratio)
