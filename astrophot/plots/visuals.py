from matplotlib.pyplot import get_cmap

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
cmap_div = get_cmap("RdBu_r")
