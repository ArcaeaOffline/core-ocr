from ..common import DeviceRoiSizes


class DeviceAutoRoiSizes(DeviceRoiSizes):
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
