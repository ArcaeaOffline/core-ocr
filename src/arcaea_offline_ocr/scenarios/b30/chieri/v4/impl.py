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
    FTR_MAX_HSV,
    FTR_MIN_HSV,
    PRS_MAX_HSV,
    PRS_MIN_HSV,
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
        self.rois.set_factor(img)

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

    def ocr_component_pfl(
        self,
        component_bgr: MatLike,
    ) -> tuple[int | None, int | None, int | None]:
        rois = self.rois.component_rois
        pure_far_lost: list[int | None] = []
        for rect in (rois.pure_rect, rois.far_rect, rois.lost_rect):
            roi = crop_xywh(component_bgr, rect.rounded())
            result = self.pfl_provider.result(roi)
            pure_far_lost.append(int(result) if result else None)

        return tuple(pure_far_lost)  # pyright: ignore[reportReturnType]

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
