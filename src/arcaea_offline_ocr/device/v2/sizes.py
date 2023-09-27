from typing import Tuple, Union


def to_int(num: Union[int, float]) -> int:
    return round(num)


def apply_factor(num: Union[int, float], factor: float):
    return num * factor


class Sizes:
    def __init__(self, factor: float):
        raise NotImplementedError()

    @property
    def TOP_BAR_HEIGHT(self):
        ...

    @property
    def SCORE_PANEL(self) -> Tuple[int, int]:
        ...

    @property
    def PFL_TOP_FROM_VMID(self):
        ...

    @property
    def PFL_LEFT_FROM_HMID(self):
        ...

    @property
    def PFL_WIDTH(self):
        ...

    @property
    def PFL_FONT_PX(self):
        ...

    @property
    def PURE_FAR_GAP(self):
        ...

    @property
    def FAR_LOST_GAP(self):
        ...

    @property
    def SCORE_BOTTOM_FROM_VMID(self):
        ...

    @property
    def SCORE_FONT_PX(self):
        ...

    @property
    def SCORE_WIDTH(self):
        ...

    @property
    def JACKET_RIGHT_FROM_HOR_MID(self):
        ...

    @property
    def JACKET_WIDTH(self):
        ...

    @property
    def MR_RT_RIGHT_FROM_HMID(self):
        ...

    @property
    def MR_RT_WIDTH(self):
        ...

    @property
    def MR_RT_HEIGHT(self):
        ...

    @property
    def TITLE_BOTTOM_FROM_VMID(self):
        ...

    @property
    def TITLE_FONT_PX(self):
        ...

    @property
    def TITLE_WIDTH_RIGHT(self):
        ...


class SizesV1(Sizes):
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
    def PFL_TOP_FROM_VMID(self):
        return self.apply_factor(135)

    @property
    def PFL_LEFT_FROM_HMID(self):
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
    def SCORE_BOTTOM_FROM_VMID(self):
        return self.apply_factor(-50)

    @property
    def SCORE_FONT_PX(self):
        return self.apply_factor(45)

    @property
    def SCORE_WIDTH(self):
        return self.apply_factor(280)

    @property
    def JACKET_RIGHT_FROM_HOR_MID(self):
        return self.apply_factor(-235)

    @property
    def JACKET_WIDTH(self):
        return self.apply_factor(375)

    @property
    def MR_RT_RIGHT_FROM_HMID(self):
        return self.apply_factor(-300)

    @property
    def MR_RT_WIDTH(self):
        return self.apply_factor(275)

    @property
    def MR_RT_HEIGHT(self):
        return self.apply_factor(75)

    @property
    def TITLE_BOTTOM_FROM_VMID(self):
        return self.apply_factor(-265)

    @property
    def TITLE_FONT_PX(self):
        return self.apply_factor(40)

    @property
    def TITLE_WIDTH_RIGHT(self):
        return self.apply_factor(275)
