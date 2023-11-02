import sqlite3
from typing import List, Union

import cv2
import numpy as np

from .types import Mat


def phash_opencv(img_gray, hash_size=8, highfreq_factor=4):
    # type: (Union[Mat, np.ndarray], int, int) -> np.ndarray
    """
    Perceptual Hash computation.

    Implementation follows
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

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


class ImagePhashDatabase:
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

            self.ids: List[str] = [
                i[0] for i in conn.execute("SELECT id FROM hashes").fetchall()
            ]
            self.hashes_byte = [
                i[0] for i in conn.execute("SELECT hash FROM hashes").fetchall()
            ]
            self.hashes = [np.frombuffer(hb, bool) for hb in self.hashes_byte]

            self.jacket_ids: List[str] = []
            self.jacket_hashes = []
            self.partner_icon_ids: List[str] = []
            self.partner_icon_hashes = []

            for _id, _hash in zip(self.ids, self.hashes):
                id_splitted = _id.split("||")
                if len(id_splitted) > 1 and id_splitted[0] == "partner_icon":
                    self.partner_icon_ids.append(id_splitted[1])
                    self.partner_icon_hashes.append(_hash)
                else:
                    self.jacket_ids.append(_id)
                    self.jacket_hashes.append(_hash)

    def calculate_phash(self, img_gray: Mat):
        return phash_opencv(
            img_gray, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor
        )

    def lookup_hash(self, image_hash: np.ndarray, *, limit: int = 5):
        image_hash = image_hash.flatten()
        xor_results = [
            (id, np.count_nonzero(image_hash ^ h))
            for id, h in zip(self.ids, self.hashes)
        ]
        return sorted(xor_results, key=lambda r: r[1])[:limit]

    def lookup_image(self, img_gray: Mat):
        image_hash = self.calculate_phash(img_gray)
        return self.lookup_hash(image_hash)[0]

    def lookup_jackets(self, img_gray: Mat, *, limit: int = 5):
        image_hash = self.calculate_phash(img_gray).flatten()
        xor_results = [
            (id, np.count_nonzero(image_hash ^ h))
            for id, h in zip(self.jacket_ids, self.jacket_hashes)
        ]
        return sorted(xor_results, key=lambda r: r[1])[:limit]

    def lookup_jacket(self, img_gray: Mat):
        return self.lookup_jackets(img_gray)[0]

    def lookup_partner_icons(self, img_gray: Mat, *, limit: int = 5):
        image_hash = self.calculate_phash(img_gray).flatten()
        xor_results = [
            (id, np.count_nonzero(image_hash ^ h))
            for id, h in zip(self.partner_icon_ids, self.partner_icon_hashes)
        ]
        return sorted(xor_results, key=lambda r: r[1])[:limit]

    def lookup_partner_icon(self, img_gray: Mat):
        return self.lookup_partner_icons(img_gray)[0]
