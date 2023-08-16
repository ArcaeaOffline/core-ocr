import io
import sqlite3
from gzip import GzipFile
from typing import Tuple

import cv2
import numpy as np

from .types import Mat


class SIFTDatabase:
    def __init__(self, db_path: str, load: bool = True):
        self.__db_path = db_path
        self.__tags = []
        self.__descriptors = []
        self.__size = None

        self.__sift = cv2.SIFT_create()
        self.__bf_matcher = cv2.BFMatcher()

        if load:
            self.load_db()

    @property
    def db_path(self):
        return self.__db_path

    @db_path.setter
    def db_path(self, value):
        self.__db_path = value

    @property
    def tags(self):
        return self.__tags

    @property
    def descriptors(self):
        return self.__descriptors

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, value: Tuple[int, int]):
        self.__size = value

    @property
    def sift(self):
        return self.__sift

    @property
    def bf_matcher(self):
        return self.__bf_matcher

    def load_db(self):
        conn = sqlite3.connect(self.db_path)
        with conn:
            cursor = conn.cursor()

            size_str = cursor.execute(
                "SELECT value FROM properties WHERE id = 'size'"
            ).fetchone()[0]
            sizr_str_arr = size_str.split(", ")
            self.size = tuple(int(s) for s in sizr_str_arr)
            tag__descriptors_bytes = cursor.execute(
                "SELECT tag, descriptors FROM sift"
            ).fetchall()

            gzipped = int(
                cursor.execute(
                    "SELECT value FROM properties WHERE id = 'gzip'"
                ).fetchone()[0]
            )
            for tag, descriptor_bytes in tag__descriptors_bytes:
                buffer = io.BytesIO(descriptor_bytes)
                self.tags.append(tag)
                if gzipped == 0:
                    self.descriptors.append(np.load(buffer))
                else:
                    gzipped_buffer = GzipFile(None, "rb", fileobj=buffer)
                    self.descriptors.append(np.load(gzipped_buffer))

    def lookup_img(
        self,
        __img: Mat,
        *,
        sift=None,
        bf=None,
    ) -> Tuple[str, float]:
        sift = sift or self.sift
        bf = bf or self.bf_matcher

        img = __img.copy()
        if self.size is not None:
            img = cv2.resize(img, self.size)
        _, descriptors = sift.detectAndCompute(img, None)

        good_results = []
        for des in self.descriptors:
            matches = bf.knnMatch(descriptors, des, k=2)
            good = sum(m.distance < 0.75 * n.distance for m, n in matches)
            good_results.append(good)
        best_match_index = max(enumerate(good_results), key=lambda i: i[1])[0]

        return (
            self.tags[best_match_index],
            good_results[best_match_index] / len(descriptors),
        )
