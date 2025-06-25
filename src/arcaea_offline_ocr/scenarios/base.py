from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, Optional

from arcaea_offline_ocr.providers import ImageIdProviderResult


@dataclass(kw_only=True)
class OcrScenarioResult:
    song_id: str
    rating_class: int
    score: int

    song_id_results: Sequence[ImageIdProviderResult] = field(default_factory=lambda: [])
    partner_id_results: Sequence[ImageIdProviderResult] = field(
        default_factory=lambda: []
    )

    pure: Optional[int] = None
    pure_inaccurate: Optional[int] = None
    pure_early: Optional[int] = None
    pure_late: Optional[int] = None
    far: Optional[int] = None
    far_inaccurate: Optional[int] = None
    far_early: Optional[int] = None
    far_late: Optional[int] = None
    lost: Optional[int] = None

    played_at: Optional[datetime] = None
    max_recall: Optional[int] = None
    clear_status: Optional[int] = None
    clear_type: Optional[int] = None
    modifier: Optional[int] = None


class OcrScenario(ABC):
    pass
