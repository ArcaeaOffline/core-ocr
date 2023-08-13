from typing import Optional

import attrs


@attrs.define
class DeviceOcrResult:
    rating_class: int
    pure: int
    far: int
    lost: int
    score: int
    max_recall: int
    song_id: Optional[str] = None
    title: Optional[str] = None
    clear_type: Optional[str] = None
