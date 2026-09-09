import cv2
import numpy as np

__all__ = ["imread_unicode"]


def imread_unicode(
    filepath: str, flags: int = cv2.IMREAD_UNCHANGED
) -> cv2.typing.MatLike:
    # https://stackoverflow.com/a/57872297/16484891
    # CC BY-SA 4.0
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)
    if img is None:
        msg = f"Failed to read image: {filepath}"
        raise OSError(msg)
    return img
