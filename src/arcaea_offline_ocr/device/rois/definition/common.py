from typing import Tuple

Rect = Tuple[int, int, int, int]


class DeviceRois:
    pure: Rect
    far: Rect
    lost: Rect
    score: Rect
    rating_class: Rect
    max_recall: Rect
    jacket: Rect
    clear_status: Rect
    partner_icon: Rect
