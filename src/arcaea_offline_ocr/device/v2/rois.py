from typing import Tuple, Union

from ...crop import crop_black_edges, crop_xywh
from ...types import Mat, XYWHRect
from .definition import DeviceV2


def to_int(num: Union[int, float]) -> int:
    return round(num)


def apply_factor(num: Union[int, float], factor: float):
    return num * factor


class Sizes:
    def __init__(self, factor: float):
        self.factor = factor

    def apply_factor(self, num):
        return apply_factor(num, self.factor)

    @property
    def TOP_BAR_HEIGHT(self):
        return self.apply_factor(50)

    @property
    def SCORE_PANEL(self) -> Tuple[int, int]:
        return tuple(self.apply_factor(num) for num in [485, 239])

    @property
    def PFL_TOP_FROM_VER_MID(self):
        return self.apply_factor(135)

    @property
    def PFL_LEFT_FROM_HOR_MID(self):
        return self.apply_factor(5)

    @property
    def PFL_WIDTH(self):
        return self.apply_factor(76)

    @property
    def PFL_FONT_PX(self):
        return self.apply_factor(26)

    @property
    def PURE_FAR_GAP(self):
        return self.apply_factor(12)

    @property
    def FAR_LOST_GAP(self):
        return self.apply_factor(10)

    @property
    def SCORE_BOTTOM_FROM_VER_MID(self):
        return self.apply_factor(-50)

    @property
    def SCORE_FONT_PX(self):
        return self.apply_factor(45)

    @property
    def SCORE_WIDTH(self):
        return self.apply_factor(280)

    @property
    def COVER_RIGHT_FROM_HOR_MID(self):
        return self.apply_factor(-235)

    @property
    def COVER_WIDTH(self):
        return self.apply_factor(375)

    @property
    def MAX_RECALL_RATING_CLASS_RIGHT_FROM_HOR_MID(self):
        return self.apply_factor(-300)

    @property
    def MAX_RECALL_RATING_CLASS_WIDTH(self):
        return self.apply_factor(275)

    @property
    def MAX_RECALL_RATING_CLASS_HEIGHT(self):
        return self.apply_factor(75)

    @property
    def TITLE_BOTTOM_FROM_VER_MID(self):
        return self.apply_factor(-265)

    @property
    def TITLE_FONT_PX(self):
        return self.apply_factor(40)

    @property
    def TITLE_WIDTH_RIGHT(self):
        return self.apply_factor(275)


class DeviceV2Rois:
    def __init__(self, device: DeviceV2, img: Mat):
        self.device = device
        self.sizes = Sizes(self.device.factor)
        self.__img = img

    @staticmethod
    def construct_int_xywh_rect(x, y, w, h) -> XYWHRect:
        return XYWHRect(*[to_int(item) for item in [x, y, w, h]])

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img: Mat):
        self.__img = (
            crop_black_edges(img) if self.device.crop_black_edges else img.copy()
        )

    @property
    def h(self):
        return self.img.shape[0]

    @property
    def ver_mid(self):
        return self.h / 2

    @property
    def w(self):
        return self.img.shape[1]

    @property
    def hor_mid(self):
        return self.w / 2

    @property
    def h_fixed(self):
        """img_height -= top_bar_height"""
        return self.h - self.sizes.TOP_BAR_HEIGHT

    @property
    def h_fixed_mid(self):
        return self.sizes.TOP_BAR_HEIGHT + self.h_fixed / 2

    @property
    def pfl_top(self):
        return self.h_fixed_mid + self.sizes.PFL_TOP_FROM_VER_MID

    @property
    def pfl_left(self):
        return self.hor_mid + self.sizes.PFL_LEFT_FROM_HOR_MID

    @property
    def pure_rect(self):
        return self.construct_int_xywh_rect(
            x=self.pfl_left,
            y=self.pfl_top,
            w=self.sizes.PFL_WIDTH,
            h=self.sizes.PFL_FONT_PX,
        )

    @property
    def pure(self):
        return crop_xywh(self.img, self.pure_rect)

    @property
    def far_rect(self):
        return self.construct_int_xywh_rect(
            x=self.pfl_left,
            y=self.pfl_top + self.sizes.PFL_FONT_PX + self.sizes.PURE_FAR_GAP,
            w=self.sizes.PFL_WIDTH,
            h=self.sizes.PFL_FONT_PX,
        )

    @property
    def far(self):
        return crop_xywh(self.img, self.far_rect)

    @property
    def lost_rect(self):
        return self.construct_int_xywh_rect(
            x=self.pfl_left,
            y=(
                self.pfl_top
                + self.sizes.PFL_FONT_PX * 2
                + self.sizes.PURE_FAR_GAP
                + self.sizes.FAR_LOST_GAP
            ),
            w=self.sizes.PFL_WIDTH,
            h=self.sizes.PFL_FONT_PX,
        )

    @property
    def lost(self):
        return crop_xywh(self.img, self.lost_rect)

    @property
    def score_rect(self):
        return self.construct_int_xywh_rect(
            x=self.hor_mid - (self.sizes.SCORE_WIDTH / 2),
            y=(
                self.h_fixed_mid
                + self.sizes.SCORE_BOTTOM_FROM_VER_MID
                - self.sizes.SCORE_FONT_PX
            ),
            w=self.sizes.SCORE_WIDTH,
            h=self.sizes.SCORE_FONT_PX,
        )

    @property
    def score(self):
        return crop_xywh(self.img, self.score_rect)

    @property
    def max_recall_rating_class_rect(self):
        x = (
            self.hor_mid
            + self.sizes.COVER_RIGHT_FROM_HOR_MID
            - self.sizes.COVER_WIDTH
            - 25
        )
        return self.construct_int_xywh_rect(
            x=x,
            y=(
                self.h_fixed_mid
                - self.sizes.SCORE_PANEL[1] / 2
                - self.sizes.MAX_RECALL_RATING_CLASS_HEIGHT
            ),
            w=self.sizes.MAX_RECALL_RATING_CLASS_WIDTH,
            h=self.sizes.MAX_RECALL_RATING_CLASS_HEIGHT,
        )

    @property
    def max_recall_rating_class(self):
        return crop_xywh(self.img, self.max_recall_rating_class_rect)

    @property
    def title_rect(self):
        return self.construct_int_xywh_rect(
            x=0,
            y=self.h_fixed_mid
            + self.sizes.TITLE_BOTTOM_FROM_VER_MID
            - self.sizes.TITLE_FONT_PX,
            w=self.hor_mid + self.sizes.TITLE_WIDTH_RIGHT,
            h=self.sizes.TITLE_FONT_PX,
        )

    @property
    def title(self):
        return crop_xywh(self.img, self.title_rect)

    @property
    def cover_rect(self):
        return self.construct_int_xywh_rect(
            x=self.hor_mid
            + self.sizes.COVER_RIGHT_FROM_HOR_MID
            - self.sizes.COVER_WIDTH,
            y=self.h_fixed_mid - self.sizes.SCORE_PANEL[1] / 2,
            w=self.sizes.COVER_WIDTH,
            h=self.sizes.COVER_WIDTH,
        )

    @property
    def cover(self):
        return crop_xywh(self.img, self.cover_rect)


class DeviceV2AutoRois(DeviceV2Rois):
    @staticmethod
    def get_factor(width: int, height: int):
        ratio = width / height
        return ((width / 16) * 9) / 720 if ratio < (16 / 9) else height / 720

    def __init__(self, img: Mat):
        factor = self.get_factor(img.shape[1], img.shape[0])
        self.sizes = Sizes(factor)
        self.__img = None
        self.img = img

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img: Mat):
        self.__img = crop_black_edges(img)
