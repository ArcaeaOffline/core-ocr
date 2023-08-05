from typing import Iterable

from attrs import define, field

from ...types import XYWHRect


def iterable_to_xywh_rect(__iter: Iterable) -> XYWHRect:
    return XYWHRect(*__iter)


@define(kw_only=True)
class DeviceV2:
    version = field(type=int)
    uuid = field(type=str)
    name = field(type=str)
    crop_black_edges = field(type=bool)
    factor = field(type=float)
    pure = field(converter=iterable_to_xywh_rect, default=[0, 0, 0, 0])
    far = field(converter=iterable_to_xywh_rect, default=[0, 0, 0, 0])
    lost = field(converter=iterable_to_xywh_rect, default=[0, 0, 0, 0])
    score = field(converter=iterable_to_xywh_rect, default=[0, 0, 0, 0])
    max_recall_rating_class = field(
        converter=iterable_to_xywh_rect, default=[0, 0, 0, 0]
    )
    title = field(converter=iterable_to_xywh_rect, default=[0, 0, 0, 0])
