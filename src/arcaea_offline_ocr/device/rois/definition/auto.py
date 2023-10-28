from .common import DeviceRois

__all__ = ["DeviceRoisAuto", "DeviceRoisAutoT1", "DeviceRoisAutoT2"]


class DeviceRoisAuto(DeviceRois):
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h


class DeviceRoisAutoT1(DeviceRoisAuto):
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
    def top_bar(self):
        return (0, 0, self.w, 50 * self.factor)

    @property
    def layout_area_h_mid(self):
        return self.h / 2 + self.top_bar[3]

    @property
    def pfl_left_from_w_mid(self):
        return 5 * self.factor

    @property
    def pfl_x(self):
        return self.w_mid + self.pfl_left_from_w_mid

    @property
    def pfl_w(self):
        return 76 * self.factor

    @property
    def pfl_h(self):
        return 26 * self.factor

    @property
    def pure(self):
        return (
            self.pfl_x,
            self.layout_area_h_mid + 110 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def far(self):
        return (
            self.pfl_x,
            self.pure[1] + self.pure[3] + 12 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def lost(self):
        return (
            self.pfl_x,
            self.far[1] + self.far[3] + 10 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def score(self):
        w = 280 * self.factor
        h = 45 * self.factor
        return (
            self.w_mid - w / 2,
            self.layout_area_h_mid - 75 * self.factor - h,
            w,
            h,
        )

    @property
    def rating_class(self):
        return (
            self.w_mid - 610 * self.factor,
            self.layout_area_h_mid - 180 * self.factor,
            265 * self.factor,
            35 * self.factor,
        )

    @property
    def max_recall(self):
        return (
            self.w_mid - 465 * self.factor,
            self.layout_area_h_mid - 215 * self.factor,
            150 * self.factor,
            35 * self.factor,
        )

    @property
    def jacket(self):
        return (
            self.w_mid - 610 * self.factor,
            self.layout_area_h_mid - 143 * self.factor,
            375 * self.factor,
            375 * self.factor,
        )

    @property
    def clear_status(self):
        w = 550 * self.factor
        h = 60 * self.factor
        return (
            self.w_mid - w / 2,
            self.layout_area_h_mid - 155 * self.factor - h,
            w * 0.4,
            h,
        )

    @property
    def partner_icon(self):
        w = 90 * self.factor
        h = 75 * self.factor
        return (self.w_mid - w / 2, 0, w, h)


class DeviceRoisAutoT2(DeviceRoisAuto):
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
    def top_bar(self):
        return (0, 0, self.w, 75 * self.factor)

    @property
    def layout_area_h_mid(self):
        return self.h / 2 + self.top_bar[3]

    @property
    def pfl_mid_from_w_mid(self):
        return 60 * self.factor

    @property
    def pfl_x(self):
        return self.w_mid + 10 * self.factor

    @property
    def pfl_w(self):
        return 100 * self.factor

    @property
    def pfl_h(self):
        return 24 * self.factor

    @property
    def pure(self):
        return (
            self.pfl_x,
            self.layout_area_h_mid + 175 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def far(self):
        return (
            self.pfl_x,
            self.pure[1] + self.pure[3] + 30 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def lost(self):
        return (
            self.pfl_x,
            self.far[1] + self.far[3] + 35 * self.factor,
            self.pfl_w,
            self.pfl_h,
        )

    @property
    def score(self):
        w = 420 * self.factor
        h = 70 * self.factor
        return (
            self.w_mid - w / 2,
            self.layout_area_h_mid - 110 * self.factor - h,
            w,
            h,
        )

    @property
    def rating_class(self):
        return (
            max(0, self.w_mid - 965 * self.factor),
            self.layout_area_h_mid - 330 * self.factor,
            350 * self.factor,
            110 * self.factor,
        )

    @property
    def max_recall(self):
        return (
            self.w_mid - 625 * self.factor,
            self.layout_area_h_mid - 275 * self.factor,
            150 * self.factor,
            50 * self.factor,
        )

    @property
    def jacket(self):
        return (
            self.w_mid - 915 * self.factor,
            self.layout_area_h_mid - 215 * self.factor,
            565 * self.factor,
            565 * self.factor,
        )

    @property
    def clear_status(self):
        w = 825 * self.factor
        h = 90 * self.factor
        return (
            self.w_mid - w / 2,
            self.layout_area_h_mid - 235 * self.factor - h,
            w * 0.4,
            h,
        )

    @property
    def partner_icon(self):
        w = 135 * self.factor
        h = 110 * self.factor
        return (self.w_mid - w / 2, 0, w, h)
