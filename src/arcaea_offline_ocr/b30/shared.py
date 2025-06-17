from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class B30OcrResultItem:
    rating_class: int
    score: int
    pure: Optional[int] = None
    far: Optional[int] = None
    lost: Optional[int] = None
    date: Optional[datetime] = None
    title: Optional[str] = None
    song_id: Optional[str] = None
