from typing import Union, Tuple

import numpy as np

from ..errors import InvalidWindow

__all__ = ("Window",)


class Window:
    def __init__(
        self,
        window: Union[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
        crpix: Tuple[float, float],
        image: "Image",
    ):
        if len(window) == 4:
            self.i_low = window[0]
            self.i_high = window[1]
            self.j_low = window[2]
            self.j_high = window[3]
        elif len(window) == 2:
            self.i_low, self.j_low = window[0]
            self.i_high, self.j_high = window[1]
        else:
            raise InvalidWindow(
                "Window must be a tuple of 4 integers or 2 tuples of 2 integers each"
            )
        self.crpix = np.asarray(crpix)
        self.image = image

    @property
    def identity(self):
        return self.image.identity

    @property
    def shape(self):
        return (self.i_high - self.i_low, self.j_high - self.j_low)

    def chunk(self, chunk_size: int):
        # number of pixels on each axis
        px = self.i_high - self.i_low
        py = self.j_high - self.j_low
        # total number of chunks desired
        chunk_tot = int(np.ceil((px * py) / chunk_size))
        # number of chunks on each axis
        cx = int(np.ceil(np.sqrt(chunk_tot * px / py)))
        cy = int(np.ceil(chunk_tot / cx))
        # number of pixels on each axis per chunk
        stepx = int(np.ceil(px / cx))
        stepy = int(np.ceil(py / cy))
        # create the windows
        windows = []
        for i in range(self.i_low, self.i_high, stepx):
            for j in range(self.j_low, self.j_high, stepy):
                i_high = min(i + stepx, self.i_high)
                j_high = min(j + stepy, self.j_high)
                windows.append(Window((i, i_high, j, j_high), self.crpix, self.image))
        return windows

    def pad(self, pad: int):
        self.i_low -= pad
        self.i_high += pad
        self.j_low -= pad
        self.j_high += pad

    def __or__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot combine Window with {type(other)}")
        new_i_low = min(self.i_low, other.i_low)
        new_i_high = max(self.i_high, other.i_high)
        new_j_low = min(self.j_low, other.j_low)
        new_j_high = max(self.j_high, other.j_high)
        return Window((new_i_low, new_i_high, new_j_low, new_j_high), self.crpix)

    def __ior__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot combine Window with {type(other)}")
        self.i_low = min(self.i_low, other.i_low)
        self.i_high = max(self.i_high, other.i_high)
        self.j_low = min(self.j_low, other.j_low)
        self.j_high = max(self.j_high, other.j_high)
        return self

    def __and__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot intersect Window with {type(other)}")
        if (
            self.i_high <= other.i_low
            or self.i_low >= other.i_high
            or self.j_high <= other.j_low
            or self.j_low >= other.j_high
        ):
            return Window(0, 0, 0, 0, self.crpix)
        new_i_low = max(self.i_low, other.i_low)
        new_i_high = min(self.i_high, other.i_high)
        new_j_low = max(self.j_low, other.j_low)
        new_j_high = min(self.j_high, other.j_high)
        return Window((new_i_low, new_i_high, new_j_low, new_j_high), self.crpix)


class Window_List:
    def __init__(self, windows: list[Window]):
        if not all(isinstance(window, Window) for window in windows):
            raise InvalidWindow(
                f"Window_List can only hold Window objects, not {tuple(type(window) for window in windows)}"
            )
        self.windows = windows

    def index(self, other: Window):
        for i, window in enumerate(self.windows):
            if other.identity == window.identity:
                return i
        else:
            raise ValueError("Could not find identity match between window list and input window")

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self):
        return len(self.windows)

    def __iter__(self):
        return iter(self.windows)
