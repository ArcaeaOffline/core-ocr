from abc import ABC
from typing import ClassVar

from typing_extensions import override

from arcaea_offline_ocr.layouts import DeviceLayout, load_device_layout
from arcaea_offline_ocr.types import XYWHRect

from .base import DeviceRois


class DeviceRoisAuto(DeviceRois, ABC):
    w: int
    h: int

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h


class _DeviceRoisFromLayout(DeviceRoisAuto, ABC):
    """Canvas-anchored fields served from a layout definition; subclasses
    provide the layout and the screen-anchored ``partner_icon``."""

    layout: ClassVar[DeviceLayout]

    @property
    @override
    def pure(self):
        return self.layout.screen_rect("pure", self.w, self.h)

    @property
    @override
    def far(self):
        return self.layout.screen_rect("far", self.w, self.h)

    @property
    @override
    def lost(self):
        return self.layout.screen_rect("lost", self.w, self.h)

    @property
    @override
    def score(self):
        return self.layout.screen_rect("score", self.w, self.h)

    @property
    @override
    def rating_class(self):
        return self.layout.screen_rect("rating_class", self.w, self.h)

    @property
    @override
    def max_recall(self):
        return self.layout.screen_rect("max_recall", self.w, self.h)

    @property
    @override
    def jacket(self):
        return self.layout.screen_rect("jacket", self.w, self.h)

    @property
    @override
    def clear_status(self):
        return self.layout.screen_rect("clear_status", self.w, self.h)


class DeviceRoisAutoT1(_DeviceRoisFromLayout):
    layout: ClassVar[DeviceLayout] = load_device_layout("t1")

    @property
    @override
    def partner_icon(self):
        # status-bar element created at runtime, pinned to the physical
        # screen top; it has no position on the layout canvas
        w = 90 * self.factor
        h = 75 * self.factor
        return XYWHRect(self.w_mid - w / 2, 0, w, h)

    # anchor helpers: partner_icon and external subclasses position
    # rois against them
    @property
    def factor(self):
        return (
            ((self.w / 16) * 9) / 720 if (self.w / self.h) < (16 / 9) else self.h / 720
        )

    @property
    def w_mid(self):
        return self.w / 2

    @property
    def h_mid(self):
        return self.h / 2

    @property
    def layout_area_h_mid(self):
        return self.h / 2 + 50 * self.factor


class DeviceRoisAutoT2(_DeviceRoisFromLayout):
    layout: ClassVar[DeviceLayout] = load_device_layout("t2")

    @property
    @override
    def partner_icon(self):
        # status-bar element created at runtime, pinned to the physical
        # screen top; it has no position on the layout canvas
        w = 135 * self.factor
        h = 110 * self.factor
        return XYWHRect(self.w_mid - w / 2, 0, w, h)

    # anchor helpers: partner_icon and external subclasses position
    # rois against them
    @property
    def factor(self):
        return (
            ((self.w / 16) * 9) / 1080
            if (self.w / self.h) < (16 / 9)
            else self.h / 1080
        )

    @property
    def w_mid(self):
        return self.w / 2

    @property
    def h_mid(self):
        return self.h / 2

    @property
    def layout_area_h_mid(self):
        return self.h / 2 + 75 * self.factor
