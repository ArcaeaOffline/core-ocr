from ....crop import crop_xywh
from ....types import Mat
from ..definition.common import DeviceRois


class DeviceRoisExtractor:
    def __init__(self, img: Mat, rois: DeviceRois):
        self.img = img
        self.sizes = rois

    def __construct_int_rect(self, rect):
        return tuple(round(r) for r in rect)

    @property
    def pure(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.pure))

    @property
    def far(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.far))

    @property
    def lost(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.lost))

    @property
    def score(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.score))

    @property
    def jacket(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.jacket))

    @property
    def rating_class(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.rating_class))

    @property
    def max_recall(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.max_recall))

    @property
    def clear_status(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.clear_status))

    @property
    def partner_icon(self):
        return crop_xywh(self.img, self.__construct_int_rect(self.sizes.partner_icon))
