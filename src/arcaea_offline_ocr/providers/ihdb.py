from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from arcaea_offline_ocr.core import hashers

from .base import ImageCategory, ImageIdProvider, ImageIdProviderResult

if TYPE_CHECKING:
    import sqlite3

    from arcaea_offline_ocr.types import Mat


T = TypeVar("T")
PROP_KEY_HASH_SIZE = "hash_size"
PROP_KEY_HIGH_FREQ_FACTOR = "high_freq_factor"
PROP_KEY_BUILT_AT = "built_at"


def _sql_hamming_distance(hash1: bytes, hash2: bytes):
    if len(hash1) != len(hash2):
        msg = "hash size does not match!"
        raise ValueError(msg)

    return sum(1 for byte1, byte2 in zip(hash1, hash2) if byte1 != byte2)


class ImageHashType(IntEnum):
    AVERAGE = 0
    DIFFERENCE = 1
    DCT = 2


@dataclass(kw_only=True)
class ImageHashDatabaseIdProviderResult(ImageIdProviderResult):
    image_hash_type: ImageHashType


class MissingPropertiesError(Exception):
    keys: list[str]

    def __init__(self, keys, *args):
        super().__init__(*args)
        self.keys = keys


class ImageHashDatabaseIdProvider(ImageIdProvider):
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.create_function("HAMMING_DISTANCE", 2, _sql_hamming_distance)

        self.properties = {
            PROP_KEY_HASH_SIZE: -1,
            PROP_KEY_HIGH_FREQ_FACTOR: -1,
            PROP_KEY_BUILT_AT: None,
        }

        self._hashes_count = {
            ImageCategory.JACKET: 0,
            ImageCategory.PARTNER_ICON: 0,
        }

        self._hash_length: int = -1

        self._initialize()

    @property
    def hash_size(self) -> int:
        return self.properties[PROP_KEY_HASH_SIZE]

    @property
    def high_freq_factor(self) -> int:
        return self.properties[PROP_KEY_HIGH_FREQ_FACTOR]

    @property
    def built_at(self) -> datetime | None:
        return self.properties.get(PROP_KEY_BUILT_AT)

    @property
    def hash_length(self):
        return self._hash_length

    def _initialize(self):
        def get_property(key, converter: Callable[[Any], T]) -> T | None:
            result = self.conn.execute(
                "SELECT value FROM properties WHERE key = ?",
                (key,),
            ).fetchone()
            return converter(result[0]) if result is not None else None

        def set_hashes_count(category: ImageCategory):
            self._hashes_count[category] = self.conn.execute(
                "SELECT COUNT(DISTINCT `id`) FROM hashes WHERE category = ?",
                (category.value,),
            ).fetchone()[0]

        properties_converter_map = {
            PROP_KEY_HASH_SIZE: lambda x: int(x),
            PROP_KEY_HIGH_FREQ_FACTOR: lambda x: int(x),
            PROP_KEY_BUILT_AT: lambda ts: datetime.fromtimestamp(
                int(ts) / 1000,
                tz=timezone.utc,
            ),
        }
        required_properties = [PROP_KEY_HASH_SIZE, PROP_KEY_HIGH_FREQ_FACTOR]

        missing_properties = []
        for property_key, converter in properties_converter_map.items():
            value = get_property(property_key, converter)
            if value is None:
                if property_key in required_properties:
                    missing_properties.append(property_key)

                continue

            self.properties[property_key] = value

        if missing_properties:
            raise MissingPropertiesError(keys=missing_properties)

        set_hashes_count(ImageCategory.JACKET)
        set_hashes_count(ImageCategory.PARTNER_ICON)

        self._hash_length = self.hash_size**2

    def lookup_hash(
        self,
        category: ImageCategory,
        hash_type: ImageHashType,
        hash_data: bytes,
    ) -> list[ImageHashDatabaseIdProviderResult]:
        cursor = self.conn.execute(
            """
SELECT
    `id`,
    HAMMING_DISTANCE(hash, ?) AS distance
FROM hashes
WHERE category = ? AND hash_type = ?
ORDER BY distance ASC LIMIT 10""",
            (hash_data, category.value, hash_type.value),
        )

        results = []
        for id_, distance in cursor.fetchall():
            results.append(
                ImageHashDatabaseIdProviderResult(
                    image_id=id_,
                    category=category,
                    confidence=(self.hash_length - distance) / self.hash_length,
                    image_hash_type=hash_type,
                ),
            )

        return results

    @staticmethod
    def hash_mat_to_bytes(hash_mat: Mat) -> bytes:
        return bytes([255 if b else 0 for b in hash_mat.flatten()])

    def results(self, img: Mat, category: ImageCategory, /):
        results: list[ImageHashDatabaseIdProviderResult] = []

        results.extend(
            self.lookup_hash(
                category,
                ImageHashType.AVERAGE,
                self.hash_mat_to_bytes(hashers.average(img, self.hash_size)),
            ),
        )
        results.extend(
            self.lookup_hash(
                category,
                ImageHashType.DIFFERENCE,
                self.hash_mat_to_bytes(hashers.difference(img, self.hash_size)),
            ),
        )
        results.extend(
            self.lookup_hash(
                category,
                ImageHashType.DCT,
                self.hash_mat_to_bytes(
                    hashers.dct(img, self.hash_size, self.high_freq_factor),
                ),
            ),
        )

        return results

    def result(
        self,
        img: Mat,
        category: ImageCategory,
        /,
        *,
        hash_type: ImageHashType = ImageHashType.DCT,
    ):
        return next(
            it for it in self.results(img, category) if it.image_hash_type == hash_type
        )
