from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from typing_extensions import override

from arcaea_offline_ocr.crop import crop_xywh
from arcaea_offline_ocr.providers import (
    ImageCategory,
    ImageIdProvider,
    OcrTextProvider,
)
from arcaea_offline_ocr.scenarios.b30.base import Best30Scenario
from arcaea_offline_ocr.scenarios.base import OcrScenarioResult

if TYPE_CHECKING:
    from cv2.typing import MatLike

from .colors import (
    BYD_MAX_HSV,
    BYD_MIN_HSV,
    FAR_BG_MAX_HSV,
    FAR_BG_MIN_HSV,
    FTR_MAX_HSV,
    FTR_MIN_HSV,
    LOST_BG_MAX_HSV,
    LOST_BG_MIN_HSV,
    PRS_MAX_HSV,
    PRS_MIN_HSV,
    PURE_BG_MAX_HSV,
    PURE_BG_MIN_HSV,
)
from .rois import ChieriBotV4Rois


class ChieriBotV4Best30Scenario(Best30Scenario):
    def __init__(
        self,
        score_provider: OcrTextProvider,
        pfl_provider: OcrTextProvider,
        image_id_provider: ImageIdProvider,
        factor: float = 1.0,
    ):
        self.__rois = ChieriBotV4Rois(factor)
        self.pfl_provider: OcrTextProvider = pfl_provider
        self.score_provider: OcrTextProvider = score_provider
        self.image_id_provider: ImageIdProvider = image_id_provider

    @property
    def rois(self):
        return self.__rois

    @property
    def factor(self):
        return self.__rois.factor

    @factor.setter
    def factor(self, factor: float):
        self.__rois.factor = factor

    def set_factor(self, img: MatLike):
        self.factor = img.shape[0] / 4400

    def ocr_component_rating_class(self, component_bgr: MatLike) -> int:
        rating_class_rect = self.rois.component_rois.rating_class_rect.rounded()

        rating_class_roi = crop_xywh(component_bgr, rating_class_rect)
        rating_class_roi = cv2.cvtColor(rating_class_roi, cv2.COLOR_BGR2HSV)
        rating_class_masks = [
            cv2.inRange(rating_class_roi, PRS_MIN_HSV, PRS_MAX_HSV),
            cv2.inRange(rating_class_roi, FTR_MIN_HSV, FTR_MAX_HSV),
            cv2.inRange(rating_class_roi, BYD_MIN_HSV, BYD_MAX_HSV),
        ]  # prs, ftr, byd only
        rating_class_results = [np.count_nonzero(m) for m in rating_class_masks]
        if max(rating_class_results) < 70:
            return 0
        return max(enumerate(rating_class_results), key=lambda i: i[1])[0] + 1

    def ocr_component_song_id_results(self, component_bgr: MatLike):
        jacket_rect = self.rois.component_rois.jacket_rect.floored()
        jacket_roi = cv2.cvtColor(
            crop_xywh(component_bgr, jacket_rect),
            cv2.COLOR_BGR2GRAY,
        )
        return self.image_id_provider.results(jacket_roi, ImageCategory.JACKET)

    def ocr_component_score(self, component_bgr: MatLike) -> int:
        score_rect = self.rois.component_rois.score_rect.rounded()
        score_roi = crop_xywh(component_bgr, score_rect)
        ocr_result = self.score_provider.result(score_roi)
        return int(ocr_result) if ocr_result else 0

    def find_pfl_rects(
        self,
        component_pfl_processed: MatLike,
    ) -> list[tuple[int, int, int, int]]:
        # sourcery skip: inline-immediately-returned-variable
        pfl_roi_find = cv2.morphologyEx(
            component_pfl_processed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, [10, 1]),
        )
        pfl_contours, _ = cv2.findContours(
            pfl_roi_find,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        pfl_rects = [cv2.boundingRect(c) for c in pfl_contours]
        pfl_rects = [
            r for r in pfl_rects if r[3] > component_pfl_processed.shape[0] * 0.1
        ]
        pfl_rects = sorted(pfl_rects, key=lambda r: r[1])
        return [
            (
                max(rect[0] - 2, 0),
                rect[1],
                min(rect[2] + 2, component_pfl_processed.shape[1]),
                rect[3],
            )
            for rect in pfl_rects
        ]

    def preprocess_component_pfl(
        self,
        component_bgr: MatLike,
    ) -> tuple[MatLike, MatLike]:
        """Return the background-filled BGR roi and a mask for locating digits."""
        pfl_rect = self.rois.component_rois.pfl_rect.rounded()
        pfl_roi = crop_xywh(component_bgr, pfl_rect)
        pfl_roi_hsv = cv2.cvtColor(pfl_roi, cv2.COLOR_BGR2HSV)

        # fill the pfl bg with background color
        bg_point = [round(i) for i in self.rois.component_rois.bg_point]
        bg_color = component_bgr[bg_point[1]][bg_point[0]]
        pure_bg_mask = cv2.inRange(pfl_roi_hsv, PURE_BG_MIN_HSV, PURE_BG_MAX_HSV)
        far_bg_mask = cv2.inRange(pfl_roi_hsv, FAR_BG_MIN_HSV, FAR_BG_MAX_HSV)
        lost_bg_mask = cv2.inRange(pfl_roi_hsv, LOST_BG_MIN_HSV, LOST_BG_MAX_HSV)
        pfl_roi[np.where(pure_bg_mask != 0)] = bg_color
        pfl_roi[np.where(far_bg_mask != 0)] = bg_color
        pfl_roi[np.where(lost_bg_mask != 0)] = bg_color

        # threshold
        pfl_roi = cv2.cvtColor(pfl_roi, cv2.COLOR_BGR2GRAY)
        # get threshold of blurred image, try ignoring the lines of bg bar
        pfl_roi_blurred = cv2.GaussianBlur(pfl_roi, (5, 5), 0)
        # pfl_roi_blurred = cv2.medianBlur(pfl_roi, 3)
        pfl_roi_blurred_threshold = cv2.threshold(
            pfl_roi_blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        # and a threshold of the original roi
        pfl_roi_threshold = cv2.threshold(
            pfl_roi,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        # turn thresholds into black background
        if pfl_roi_blurred_threshold[2][2] == 255:
            pfl_roi_blurred_threshold = 255 - pfl_roi_blurred_threshold
        if pfl_roi_threshold[2][2] == 255:
            pfl_roi_threshold = 255 - pfl_roi_threshold
        # return a bitwise_and result
        result = cv2.bitwise_and(pfl_roi_blurred_threshold, pfl_roi_threshold)
        result_eroded = cv2.erode(
            result,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        )
        mask = result_eroded if len(self.find_pfl_rects(result_eroded)) == 3 else result
        return pfl_roi, mask

    def ocr_component_pfl(
        self,
        component_bgr: MatLike,
    ) -> tuple[int | None, int | None, int | None]:
        try:
            pfl_roi, pfl_mask = self.preprocess_component_pfl(component_bgr)
            pfl_rects = self.find_pfl_rects(pfl_mask)
            pure_far_lost: list[int | None] = []
            for pfl_roi_rect in pfl_rects:
                # feed CRNN the BGR roi, the mask is only for locating digits
                roi = crop_xywh(pfl_roi, pfl_roi_rect)
                result = self.pfl_provider.result(roi)
                pure_far_lost.append(int(result) if result else None)

            return tuple(pure_far_lost)  # pyright: ignore[reportReturnType]
        except Exception:  # noqa: BLE001
            return (None, None, None)

    def ocr_component(self, component_bgr: MatLike) -> OcrScenarioResult:
        component_blur = cv2.GaussianBlur(component_bgr, (5, 5), 0)
        rating_class = self.ocr_component_rating_class(component_blur)
        song_id_results = self.ocr_component_song_id_results(component_bgr)
        score = self.ocr_component_score(component_bgr)
        pure, far, lost = self.ocr_component_pfl(component_bgr)
        return OcrScenarioResult(
            song_id=song_id_results[0].image_id,
            song_id_results=song_id_results,
            rating_class=rating_class,
            score=score,
            pure=pure,
            far=far,
            lost=lost,
            played_at=None,
        )

    @override
    def components(self, img: MatLike, /):
        """
        :param img: BGR format image
        """
        self.set_factor(img)
        return self.rois.components(img)

    @override
    def result(self, component_img: MatLike, /):
        return self.ocr_component(component_img)

    @override
    def results(self, img: MatLike, /) -> list[OcrScenarioResult]:
        """
        :param img: BGR format image
        """
        return [self.ocr_component(component) for component in self.components(img)]
