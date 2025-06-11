from typing import Union, Tuple

import numpy as np

from ..errors import InvalidWindow

__all__ = ("Window",)


class Window:
    def __init__(
        self,
        window: Union[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
        crpix: Tuple[int, int],
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
        self.crpix = np.asarray(crpix, dtype=int)
        self.image = image

    def get_indices(self, crpix: tuple[int, int] = None):
        if crpix is None:
            crpix = self.crpix
        shift = crpix - self.crpix
        return slice(self.i_low - shift[0], self.i_high - shift[0]), slice(
            self.j_low - shift[1], self.j_high - shift[1]
        )

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
