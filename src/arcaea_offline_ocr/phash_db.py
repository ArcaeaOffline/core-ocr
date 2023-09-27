import sqlite3

import imagehash
import numpy as np
from PIL import Image


def hamming_distance_sql_function(user_input, db_entry) -> int:
    return np.count_nonzero(
        np.frombuffer(user_input, bool) ^ np.frombuffer(db_entry, bool)
    )


class ImagePHashDatabase:
    def __init__(self, db_path: str):
        with sqlite3.connect(db_path) as conn:
            self.hash_size = int(
                conn.execute(
                    "SELECT value FROM properties WHERE key = 'hash_size'"
                ).fetchone()[0]
            )
            self.highfreq_factor = int(
                conn.execute(
                    "SELECT value FROM properties WHERE key = 'highfreq_factor'"
                ).fetchone()[0]
            )
            self.built_timestamp = int(
                conn.execute(
                    "SELECT value FROM properties WHERE key = 'built_timestamp'"
                ).fetchone()[0]
            )

            # self.conn.create_function(
            #     "HAMMING_DISTANCE",
            #     2,
            #     hamming_distance_sql_function,
            #     deterministic=True,
            # )

            self.ids = [i[0] for i in conn.execute("SELECT id FROM hashes").fetchall()]
            self.hashes_byte = [
                i[0] for i in conn.execute("SELECT hash FROM hashes").fetchall()
            ]
            self.hashes = [np.frombuffer(hb, bool) for hb in self.hashes_byte]
            self.hashes_slice_size = round(len(self.hashes_byte[0]) * 0.25)
            self.hashes_head = [h[: self.hashes_slice_size] for h in self.hashes]
            self.hashes_tail = [h[-self.hashes_slice_size :] for h in self.hashes]

    def lookup_hash(self, image_hash: imagehash.ImageHash, *, limit: int = 5):
        image_hash = image_hash.hash.flatten()
        # image_hash_head = image_hash[: self.hashes_slice_size]
        # image_hash_tail = image_hash[-self.hashes_slice_size :]
        # head_xor_results = [image_hash_head ^ h for h in self.hashes]
        # tail_xor_results = [image_hash_head ^ h for h in self.hashes]
        xor_results = [
            (id, np.count_nonzero(image_hash ^ h))
            for id, h in zip(self.ids, self.hashes)
        ]
        return sorted(xor_results, key=lambda r: r[1])[:limit]

    def lookup_image(self, pil_image: Image.Image):
        image_hash = imagehash.phash(
            pil_image, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor
        )
        return self.lookup_hash(image_hash)[0]
