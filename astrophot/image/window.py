from typing import Union, Tuple, List

import numpy as np

from ..errors import InvalidWindow

__all__ = ("Window",)


class Window:
    def __init__(
        self,
        window: Union[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
        image: "Image",
    ):
        self.extent = window
        self.image = image

    @property
    def identity(self):
        return self.image.identity

    @property
    def crpix(self):
        return self.image.crpix

    @property
    def shape(self):
        return (self.i_high - self.i_low, self.j_high - self.j_low)

    @property
    def extent(self):
        return (self.i_low, self.i_high, self.j_low, self.j_high)

    @extent.setter
    def extent(
        self, value: Union[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        if len(value) == 4:
            self.i_low, self.i_high, self.j_low, self.j_high = value
        elif len(value) == 2:
            self.i_low, self.j_low = value[0]
            self.i_high, self.j_high = value[1]
        else:
            raise ValueError(
                "Extent must be formatted as (i_low, i_high, j_low, j_high) or ((i_low, j_low), (i_high, j_high))"
            )

    def chunk(self, chunk_size: int) -> List["Window"]:
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
                windows.append(Window((i, i_high, j, j_high), self.image))
        return windows

    def pad(self, pad: int):
        self.i_low -= pad
        self.i_high += pad
        self.j_low -= pad
        self.j_high += pad

    def copy(self):
        return Window((self.i_low, self.i_high, self.j_low, self.j_high), self.image)

    def __or__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot combine Window with {type(other)}")
        if self.image != other.image:
            raise InvalidWindow(
                f"Cannot combine Windows from different images: {self.image.identity} and {other.image.identity}"
            )
        new_i_low = min(self.i_low, other.i_low)
        new_i_high = max(self.i_high, other.i_high)
        new_j_low = min(self.j_low, other.j_low)
        new_j_high = max(self.j_high, other.j_high)
        return Window((new_i_low, new_i_high, new_j_low, new_j_high), self.image)

    def __ior__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot combine Window with {type(other)}")
        if self.image != other.image:
            raise InvalidWindow(
                f"Cannot combine Windows from different images: {self.image.identity} and {other.image.identity}"
            )
        self.i_low = min(self.i_low, other.i_low)
        self.i_high = max(self.i_high, other.i_high)
        self.j_low = min(self.j_low, other.j_low)
        self.j_high = max(self.j_high, other.j_high)
        return self

    def __and__(self, other: "Window"):
        if not isinstance(other, Window):
            raise TypeError(f"Cannot intersect Window with {type(other)}")
        if self.image.identity != other.image.identity:
            raise InvalidWindow(
                f"Cannot combine Windows from different images: {self.image.identity} and {other.image.identity}"
            )
        if (
            self.i_high <= other.i_low
            or self.i_low >= other.i_high
            or self.j_high <= other.j_low
            or self.j_low >= other.j_high
        ):
            return Window((0, 0, 0, 0), self.image)
        # fixme handle crpix
        new_i_low = max(self.i_low, other.i_low)
        new_i_high = min(self.i_high, other.i_high)
        new_j_low = max(self.j_low, other.j_low)
        new_j_high = min(self.j_high, other.j_high)
        return Window((new_i_low, new_i_high, new_j_low, new_j_high), self.image)

    def __str__(self):
        return f"Window({self.i_low}, {self.i_high}, {self.j_low}, {self.j_high})"


class WindowList:
    def __init__(self, windows: list[Window]):
        if not all(isinstance(window, Window) for window in windows):
            raise InvalidWindow(
                f"Window_List can only hold Window objects, not {tuple(type(window) for window in windows)}"
            )
        self.windows = windows

    def index(self, other: Window) -> int:
        for i, window in enumerate(self.windows):
            if other.identity == window.identity:
                return i
        else:
            raise IndexError("Could not find identity match between window list and input window")

    def __and__(self, other: "WindowList"):
        if not isinstance(other, WindowList):
            raise TypeError(f"Cannot intersect WindowList with {type(other)}")
        if len(self.windows) == 0 or len(other.windows) == 0:
            return WindowList([])
        new_windows = []
        for other_window in other.windows:
            try:
                i = self.index(other_window)
            except IndexError:
                continue  # skip if the window is not in self.windows
            new_windows.append(self.windows[i] & other_window)
        return WindowList(new_windows)

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self):
        return len(self.windows)

    def __iter__(self):
        return iter(self.windows)
