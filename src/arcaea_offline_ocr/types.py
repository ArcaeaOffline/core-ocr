from typing import NamedTuple

import numpy as np

# from pylance
Mat = np.ndarray[int, np.dtype[np.generic]]


class XYWHRect(NamedTuple):
    x: int
    y: int
    w: int
    h: int
