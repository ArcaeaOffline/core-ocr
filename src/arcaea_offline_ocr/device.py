from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(kw_only=True)
class Device:
    version: int
    uuid: str
    name: str
    pure: Tuple[int, int, int, int]
    far: Tuple[int, int, int, int]
    lost: Tuple[int, int, int, int]
    max_recall: Tuple[int, int, int, int]
    rating_class: Tuple[int, int, int, int]
    score: Tuple[int, int, int, int]
    title: Tuple[int, int, int, int]

    @classmethod
    def from_json_object(cls, json_dict: Dict[str, Any]):
        if json_dict["version"] == 1:
            return cls(
                version=1,
                uuid=json_dict["uuid"],
                name=json_dict["name"],
                pure=json_dict["pure"],
                far=json_dict["far"],
                lost=json_dict["lost"],
                max_recall=json_dict["max_recall"],
                rating_class=json_dict["rating_class"],
                score=json_dict["score"],
                title=json_dict["title"],
            )
