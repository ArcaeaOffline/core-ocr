from arcaea_offline_ocr.crop import crop_xywh
from arcaea_offline_ocr.scenarios.device.rois import DeviceRois
from arcaea_offline_ocr.types import Mat


class DeviceRoisExtractor:
    def __init__(self, img: Mat, rois: DeviceRois):
        self.img = img
        self.sizes = rois

    @property
    def pure(self):
        return crop_xywh(self.img, self.sizes.pure.rounded())

    @property
    def far(self):
        return crop_xywh(self.img, self.sizes.far.rounded())

    @property
    def lost(self):
        return crop_xywh(self.img, self.sizes.lost.rounded())

    @property
    def score(self):
        return crop_xywh(self.img, self.sizes.score.rounded())

    @property
    def jacket(self):
        return crop_xywh(self.img, self.sizes.jacket.rounded())

    @property
    def rating_class(self):
        return crop_xywh(self.img, self.sizes.rating_class.rounded())

    @property
    def max_recall(self):
        return crop_xywh(self.img, self.sizes.max_recall.rounded())

    @property
    def clear_status(self):
        return crop_xywh(self.img, self.sizes.clear_status.rounded())

    @property
    def partner_icon(self):
        return crop_xywh(self.img, self.sizes.partner_icon.rounded())
