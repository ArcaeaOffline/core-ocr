import sqlite3

import cv2
import numpy as np


def phash_opencv(img_gray, hash_size=8, highfreq_factor=4):
    # type: (cv2.Mat | np.ndarray, int, int) -> np.ndarray
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    Adapted from `imagehash.phash`, pure opencv implementation

    The result is slightly different from `imagehash.phash`.
    """
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    img_size = hash_size * highfreq_factor
    image = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    image = np.float32(image)
    dct = cv2.dct(image)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff


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

    def lookup_hash(self, image_hash: np.ndarray, *, limit: int = 5):
        image_hash = image_hash.flatten()
        # image_hash_head = image_hash[: self.hashes_slice_size]
        # image_hash_tail = image_hash[-self.hashes_slice_size :]
        # head_xor_results = [image_hash_head ^ h for h in self.hashes]
        # tail_xor_results = [image_hash_head ^ h for h in self.hashes]
        xor_results = [
            (id, np.count_nonzero(image_hash ^ h))
            for id, h in zip(self.ids, self.hashes)
        ]
        return sorted(xor_results, key=lambda r: r[1])[:limit]

    def lookup_image(self, img_gray: cv2.Mat):
        image_hash = phash_opencv(
            img_gray, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor
        )
        return self.lookup_hash(image_hash)[0]
