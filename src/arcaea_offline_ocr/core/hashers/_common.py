import cv2


def resize_image(src: cv2.typing.MatLike, dsize: ...) -> cv2.typing.MatLike:
    return cv2.resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_AREA)
