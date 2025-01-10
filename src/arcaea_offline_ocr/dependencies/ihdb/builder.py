import logging
from datetime import datetime, timezone
from sqlite3 import Connection
from typing import List

from arcaea_offline_ocr.core import hashers

from .index import ImageHashesDatabase
from .models import ImageHash, ImageHashBuildTask, ImageHashHashType

logger = logging.getLogger(__name__)


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
        tasks: List[ImageHashBuildTask],
        *,
        hash_size: int = 16,
        high_freq_factor: int = 4,
    ):
        rows: List[ImageHash] = []

        for task in tasks:
            try:
                img_gray = task.imread_function(task.image_path)

                for hash_type, hash_mat in [
                    (
                        ImageHashHashType.AVERAGE,
                        hashers.average(img_gray, hash_size),
                    ),
                    (
                        ImageHashHashType.DCT,
                        hashers.dct(img_gray, hash_size, high_freq_factor),
                    ),
                    (
                        ImageHashHashType.DIFFERENCE,
                        hashers.difference(img_gray, hash_size),
                    ),
                ]:
                    rows.append(
                        ImageHash(
                            hash_type=hash_type,
                            type=task.type,
                            label=task.label,
                            hash=ImageHashesDatabase.hash_mat_to_bytes(hash_mat),
                        )
                    )
            except Exception:
                logger.exception("Error processing task %r", task)

        conn.execute("CREATE TABLE properties (`key` VARCHAR, `value` VARCHAR)")
        conn.execute(
            "CREATE TABLE hashes (`hash_type` INTEGER, `type` INTEGER, `label` VARCHAR, `hash` BLOB)"
        )

        now = datetime.now(tz=timezone.utc)
        timestamp = int(now.timestamp() * 1000)

        cls.__insert_property(conn, ImageHashesDatabase.KEY_HASH_SIZE, str(hash_size))
        cls.__insert_property(
            conn, ImageHashesDatabase.KEY_HIGH_FREQ_FACTOR, str(high_freq_factor)
        )
        cls.__insert_property(
            conn, ImageHashesDatabase.KEY_BUILT_TIMESTAMP, str(timestamp)
        )

        conn.executemany(
            "INSERT INTO hashes (hash_type, type, label, hash) VALUES (?, ?, ?, ?)",
            [
                (row.hash_type.value, row.type.value, row.label, row.hash)
                for row in rows
            ],
        )
        conn.commit()
