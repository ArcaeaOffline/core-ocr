from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arcaea_offline_ocr.crop import crop_xywh
from arcaea_offline_ocr.layouts import CardLayout, load_card_layout

if TYPE_CHECKING:
    from cv2.typing import MatLike


class ChieriBotV4ComponentRois:
    layout: ClassVar[CardLayout] = load_card_layout("chieri_v4")

    def __init__(self, factor: float = 1.0):
        self.__factor = factor

    @property
    def factor(self):
        return self.__factor

    @factor.setter
    def factor(self, factor: float):
        self.__factor = factor

    @property
    def rating_class_rect(self):
        return self.layout.field_rect("rating_class", self.factor)

    @property
    def jacket_rect(self):
        return self.layout.field_rect("jacket", self.factor)

    @property
    def score_rect(self):
        return self.layout.field_rect("score", self.factor)

    @property
    def pure_rect(self):
        return self.layout.field_rect("pure", self.factor)

    @property
    def far_rect(self):
        return self.layout.field_rect("far", self.factor)

    @property
    def lost_rect(self):
        return self.layout.field_rect("lost", self.factor)


class ChieriBotV4Rois:
    layout: ClassVar[CardLayout] = load_card_layout("chieri_v4")

    def __init__(self, factor: float = 1.0):
        self.__factor = factor
        self.__component_rois = ChieriBotV4ComponentRois(factor)

    @property
    def component_rois(self):
        return self.__component_rois

    @property
    def factor(self):
        return self.__factor

    @factor.setter
    def factor(self, factor: float):
        self.__factor = factor
        self.__component_rois.factor = factor

    def set_factor(self, img_bgr: MatLike):
        self.factor = self.layout.factor(img_bgr.shape[1])

    def components(self, img_bgr: MatLike) -> list[MatLike]:
        return [
            crop_xywh(img_bgr, self.layout.card_rect(i, self.factor).rounded())
            for i in range(self.layout.component_count)
        ]
