from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceOcrResult:
    rating_class: int
    pure: int
    far: int
    lost: int
    score: int
    max_recall: Optional[int] = None
    song_id: Optional[str] = None
    song_id_possibility: Optional[float] = None
    clear_status: Optional[int] = None
    partner_id: Optional[str] = None
    partner_id_possibility: Optional[float] = None
