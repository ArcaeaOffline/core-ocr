from cv2 import IMREAD_UNCHANGED, imdecode
from numpy import fromfile as np_fromfile
from numpy import uint8

from .types import Mat


def imread_unicode(filepath: str) -> Mat:
    # https://stackoverflow.com/a/57872297/16484891
    # CC BY-SA 4.0
    return imdecode(np_fromfile(filepath, dtype=uint8), IMREAD_UNCHANGED)
