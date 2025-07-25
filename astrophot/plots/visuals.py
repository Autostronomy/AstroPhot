from matplotlib.pyplot import get_cmap

# from matplotlib.colors import ListedColormap
# import numpy as np

__all__ = ["main_pallet", "cmap_grad", "cmap_div"]

main_pallet = {
    "primary1": "tab:blue",
    "primary2": "tab:orange",
    "primary3": "tab:red",
    "secondary1": "tab:green",
    "secondary2": "tab:purple",
    "pop": "tab:pink",
}

cmap_grad = get_cmap("inferno")
cmap_div = get_cmap("seismic")  # twilight  RdBu_r
# print(__file__)
# colors = np.load(f"{__file__[:-10]}/managua_cmap.npy")
# cmap_div = ListedColormap(list(reversed(colors)), name="mangua")
