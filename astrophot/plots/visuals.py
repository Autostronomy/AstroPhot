from matplotlib.pyplot import get_cmap

__all__ = ["main_pallet", "cmap_grad", "cmap_div"]

main_pallet = {
    "primary1": "tab:green",
    "primary2": "limegreen",
    "primary3": "lime",
    "secondary1": "tab:blue",
    "secondary2": "blue",
    "pop": "tab:orange",
}

cmap_grad = get_cmap("inferno")
cmap_div = get_cmap("seismic")
