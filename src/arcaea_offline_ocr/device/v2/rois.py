from typing import Union

from ...crop import crop_black_edges, crop_xywh
from ...types import Mat, XYWHRect
from .definition import DeviceV2
from .sizes import Sizes, SizesV1


def to_int(num: Union[int, float]) -> int:
    return round(num)


class DeviceV2Rois:
    def __init__(self, device: DeviceV2, img: Mat):
        self.device = device
        self.sizes = SizesV1(self.device.factor)
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
    def vmid(self):
        return self.h / 2

    @property
    def w(self):
        return self.img.shape[1]

    @property
    def hmid(self):
        return self.w / 2

    @property
    def h_without_top_bar(self):
        """img_height -= top_bar_height"""
        return self.h - self.sizes.TOP_BAR_HEIGHT

    @property
    def h_without_top_bar_mid(self):
        return self.sizes.TOP_BAR_HEIGHT + self.h_without_top_bar / 2

    @property
    def pfl_top(self):
        return self.h_without_top_bar_mid + self.sizes.PFL_TOP_FROM_VMID

    @property
    def pfl_left(self):
        return self.hmid + self.sizes.PFL_LEFT_FROM_HMID

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
            x=self.hmid - (self.sizes.SCORE_WIDTH / 2),
            y=(
                self.h_without_top_bar_mid
                + self.sizes.SCORE_BOTTOM_FROM_VMID
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
            self.hmid
            + self.sizes.JACKET_RIGHT_FROM_HOR_MID
            - self.sizes.JACKET_WIDTH
            - 25 * self.sizes.factor
        )
        return self.construct_int_xywh_rect(
            x=x,
            y=(
                self.h_without_top_bar_mid
                - self.sizes.SCORE_PANEL[1] / 2
                - self.sizes.MR_RT_HEIGHT
            ),
            w=self.sizes.MR_RT_WIDTH,
            h=self.sizes.MR_RT_HEIGHT,
        )

    @property
    def max_recall_rating_class(self):
        return crop_xywh(self.img, self.max_recall_rating_class_rect)

    @property
    def title_rect(self):
        return self.construct_int_xywh_rect(
            x=0,
            y=self.h_without_top_bar_mid
            + self.sizes.TITLE_BOTTOM_FROM_VMID
            - self.sizes.TITLE_FONT_PX,
            w=self.hmid + self.sizes.TITLE_WIDTH_RIGHT,
            h=self.sizes.TITLE_FONT_PX,
        )

    @property
    def title(self):
        return crop_xywh(self.img, self.title_rect)

    @property
    def jacket_rect(self):
        return self.construct_int_xywh_rect(
            x=self.hmid
            + self.sizes.JACKET_RIGHT_FROM_HOR_MID
            - self.sizes.JACKET_WIDTH,
            y=self.h_without_top_bar_mid - self.sizes.SCORE_PANEL[1] / 2,
            w=self.sizes.JACKET_WIDTH,
            h=self.sizes.JACKET_WIDTH,
        )

    @property
    def jacket(self):
        return crop_xywh(self.img, self.jacket_rect)


class DeviceV2AutoRois(DeviceV2Rois):
    @staticmethod
    def get_factor(width: int, height: int):
        ratio = width / height
        return ((width / 16) * 9) / 720 if ratio < (16 / 9) else height / 720

    def __init__(self, img: Mat):
        factor = self.get_factor(img.shape[1], img.shape[0])
        self.sizes = SizesV1(factor)
        self.__img = None
        self.img = img

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img: Mat):
        self.__img = crop_black_edges(img)
