from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import cv2

from arcaea_offline_ocr.core import hashers
from arcaea_offline_ocr.providers.ihdb import (
    PROP_KEY_BUILT_AT,
    PROP_KEY_HASH_SIZE,
    PROP_KEY_HIGH_FREQ_FACTOR,
    ImageHashDatabaseIdProvider,
    ImageHashType,
)

if TYPE_CHECKING:
    from sqlite3 import Connection

    from arcaea_offline_ocr.providers import ImageCategory
    from arcaea_offline_ocr.types import Mat


def _default_imread_gray(image_path: str):
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)


@dataclass
class ImageHashDatabaseBuildTask:
    image_path: str
    image_id: str
    category: ImageCategory
    imread_function: Callable[[str], Mat] = _default_imread_gray


@dataclass
class _ImageHash:
    image_id: str
    category: ImageCategory
    image_hash_type: ImageHashType
    hash: bytes


class ImageHashesDatabaseBuilder:
    @staticmethod
    def __insert_property(conn: Connection, key: str, value: str):
        return conn.execute(
            "INSERT INTO properties (key, value) VALUES (?, ?)",
            (key, value),
        )

    @classmethod
    def build(
        cls,
        conn: Connection,
        tasks: list[ImageHashDatabaseBuildTask],
        *,
        hash_size: int = 16,
        high_freq_factor: int = 4,
    ):
        hashes: list[_ImageHash] = []

        for task in tasks:
            img_gray = task.imread_function(task.image_path)

            for hash_type, hash_mat in [
                (
                    ImageHashType.AVERAGE,
                    hashers.average(img_gray, hash_size),
                ),
                (
                    ImageHashType.DCT,
                    hashers.dct(img_gray, hash_size, high_freq_factor),
                ),
                (
                    ImageHashType.DIFFERENCE,
                    hashers.difference(img_gray, hash_size),
                ),
            ]:
                hashes.append(
                    _ImageHash(
                        image_id=task.image_id,
                        image_hash_type=hash_type,
                        category=task.category,
                        hash=ImageHashDatabaseIdProvider.hash_mat_to_bytes(hash_mat),
                    ),
                )

        conn.execute("CREATE TABLE properties (`key` VARCHAR, `value` VARCHAR)")
        conn.execute(
            """CREATE TABLE hashes (
`id` VARCHAR,
`category` INTEGER,
`hash_type` INTEGER,
`hash` BLOB
)""",
        )

        now = datetime.now(tz=timezone.utc)
        timestamp = int(now.timestamp() * 1000)

        cls.__insert_property(conn, PROP_KEY_HASH_SIZE, str(hash_size))
        cls.__insert_property(conn, PROP_KEY_HIGH_FREQ_FACTOR, str(high_freq_factor))
        cls.__insert_property(conn, PROP_KEY_BUILT_AT, str(timestamp))

        conn.executemany(
            """INSERT INTO hashes (`id`, `category`, `hash_type`, `hash`)
                            VALUES (?, ?, ?, ?)""",
            [
                (it.image_id, it.category.value, it.image_hash_type.value, it.hash)
                for it in hashes
            ],
        )
        conn.commit()
