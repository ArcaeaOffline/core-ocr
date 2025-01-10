import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, TypeVar

from arcaea_offline_ocr.core import hashers
from arcaea_offline_ocr.types import Mat

from .models import ImageHashHashType, ImageHashResult, ImageHashType

T = TypeVar("T")


def _sql_hamming_distance(hash1: bytes, hash2: bytes):
    assert len(hash1) == len(hash2), "hash size does not match!"
    count = sum(1 for byte1, byte2 in zip(hash1, hash2) if byte1 != byte2)
    return count


class ImageHashesDatabasePropertyMissingError(Exception):
    pass


class ImageHashesDatabase:
    KEY_HASH_SIZE = "hash_size"
    KEY_HIGH_FREQ_FACTOR = "high_freq_factor"
    KEY_BUILT_TIMESTAMP = "built_timestamp"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.create_function("HAMMING_DISTANCE", 2, _sql_hamming_distance)

        self._hash_size: int = -1
        self._high_freq_factor: int = -1
        self._built_time: Optional[datetime] = None

        self._hashes_count = {
            ImageHashType.JACKET: 0,
            ImageHashType.PARTNER_ICON: 0,
        }

        self._hash_length: int = -1

        self._initialize()

    @property
    def hash_size(self):
        return self._hash_size

    @property
    def high_freq_factor(self):
        return self._high_freq_factor

    @property
    def hash_length(self):
        return self._hash_length

    def _initialize(self):
        def query_property(key, convert_func: Callable[[Any], T]) -> Optional[T]:
            result = self.conn.execute(
                "SELECT value FROM properties WHERE key = ?",
                (key,),
            ).fetchone()
            return convert_func(result[0]) if result is not None else None

        def set_hashes_count(type: ImageHashType):
            self._hashes_count[type] = self.conn.execute(
                "SELECT COUNT(DISTINCT label) FROM hashes WHERE type = ?", (type.value,)
            ).fetchone()[0]

        hash_size = query_property(self.KEY_HASH_SIZE, lambda x: int(x))
        if hash_size is None:
            raise ImageHashesDatabasePropertyMissingError("hash_size")
        self._hash_size = hash_size

        high_freq_factor = query_property(self.KEY_HIGH_FREQ_FACTOR, lambda x: int(x))
        if high_freq_factor is None:
            raise ImageHashesDatabasePropertyMissingError("high_freq_factor")
        self._high_freq_factor = high_freq_factor

        self._built_time = query_property(
            self.KEY_BUILT_TIMESTAMP,
            lambda ts: datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc),
        )

        set_hashes_count(ImageHashType.JACKET)
        set_hashes_count(ImageHashType.PARTNER_ICON)

        self._hash_length = self._hash_size**2

    def lookup_hash(
        self, type: ImageHashType, hash_type: ImageHashHashType, hash: bytes
    ) -> List[ImageHashResult]:
        cursor = self.conn.execute(
            "SELECT"
            " label,"
            " HAMMING_DISTANCE(hash, ?) AS distance"
            " FROM hashes"
            " WHERE type = ? AND hash_type = ?"
            " ORDER BY distance ASC LIMIT 10",
            (hash, type.value, hash_type.value),
        )

        results = []
        for label, distance in cursor.fetchall():
            results.append(
                ImageHashResult(
                    hash_type=hash_type,
                    type=type,
                    label=label,
                    confidence=(self.hash_length - distance) / self.hash_length,
                )
            )

        return results

    @staticmethod
    def hash_mat_to_bytes(hash: Mat) -> bytes:
        return bytes([255 if b else 0 for b in hash.flatten()])

    def identify_image(self, type: ImageHashType, img) -> List[ImageHashResult]:
        results = []

        ahash = hashers.average(img, self.hash_size)
        dhash = hashers.difference(img, self.hash_size)
        phash = hashers.dct(img, self.hash_size, self.high_freq_factor)

        results.extend(
            self.lookup_hash(
                type, ImageHashHashType.AVERAGE, self.hash_mat_to_bytes(ahash)
            )
        )
        results.extend(
            self.lookup_hash(
                type, ImageHashHashType.DIFFERENCE, self.hash_mat_to_bytes(dhash)
            )
        )
        results.extend(
            self.lookup_hash(type, ImageHashHashType.DCT, self.hash_mat_to_bytes(phash))
        )

        return results
