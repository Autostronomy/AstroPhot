import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = ["main_pallet", "cmap_grad", "cmap_div"]

main_pallet = {
    "primary1": "#5FAD41",
    "primary2": "#46A057",
    "primary3": "#2D936C",
    "secondary1": "#595122",
    "secondary2": "#BFAE48",
    "pop": "#391463",    
}

grad_list = [
    "#000000",
    "#1A1F16",
    "#1E3F20",
    "#335E31", #"#294C28",
    "#477641", #"#345830",
    "#5D986D", #"#4A7856",
    "#88BF9E", #"#6FB28A",
    "#94ECBE",
    "#FFFFFF"
]
#grad_list = ["#000000", "#1A1F16", "#1E3F20", "#294C28", "#345830", "#4A7856", "#6FB28A", "#94ECBE", "#FFFFFF"]
grad_cdict = {"red": [], "green": [], "blue": []}
cpoints = np.linspace(0, 1, len(grad_list))
for i in range(len(grad_list)):
    grad_cdict["red"].append(
        [cpoints[i], int(grad_list[i][1:3], 16) / 256, int(grad_list[i][1:3], 16) / 256]
    )
    grad_cdict["green"].append(
        [cpoints[i], int(grad_list[i][3:5], 16) / 256, int(grad_list[i][3:5], 16) / 256]
    )
    grad_cdict["blue"].append(
        [cpoints[i], int(grad_list[i][5:7], 16) / 256, int(grad_list[i][5:7], 16) / 256]
    )
cmap_grad = LinearSegmentedColormap("cmap_grad", grad_cdict)

div_list = [
    "#332A1F",
    "#514129",
    "#7C6527",
    "#A2862A",
    "#DAB944",
    "#FFFFFF",
    "#7EC87E",
    "#3EA343",
    "#267D2F",
    "#0D5D09",
    "#073805",
]
#div_list = ["#083D77", "#7E886B", "#B9AE65", "#FFFFFF", "#F1B555", "#EE964B", "#F95738"]
div_cdict = {"red": [], "green": [], "blue": []}
cpoints = np.linspace(0, 1, len(div_list))
for i in range(len(div_list)):
    div_cdict["red"].append(
        [cpoints[i], int(div_list[i][1:3], 16) / 256, int(div_list[i][1:3], 16) / 256]
    )
    div_cdict["green"].append(
        [cpoints[i], int(div_list[i][3:5], 16) / 256, int(div_list[i][3:5], 16) / 256]
    )
    div_cdict["blue"].append(
        [cpoints[i], int(div_list[i][5:7], 16) / 256, int(div_list[i][5:7], 16) / 256]
    )
cmap_div = LinearSegmentedColormap("cmap_div", div_cdict)

# P = plt.cm.plasma_r
# C = plt.cm.cividis
# N = 3
# print(np.concatenate((C(np.linspace(0,1,N)),np.array(((1,1,1,1),)),P(np.linspace(0,1,N)))))
# cmap_div = ListedColormap(["#083D77", "#7E886B", "#B9AE65", "#FFFFFF", "#F1B555", "#EE964B", "#F95738"])
    
# main_pallet = {
#     "primary1": "g",
#     "primary2": "r",
#     "primary3": "b",
#     "primary4": "ornnge",
#     "primary5": "cyan",
#     "secondary1": "purple",
#     "secondary2": "salmon",
#     "secondary3": "k",
#     "pop": "yellow",
# }

# cmap_grad = plt.cm.magma
# cmap_div = plt.cm.seismic

# from matplotlib.colors import LinearSegmentedColormap
# cmaplist = ["#000000", "#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"]
# cdict = {"red": [], "green": [], "blue": []}
# cpoints = np.linspace(0, 1, len(cmaplist))
# for i in range(len(cmaplist)):
#     cdict["red"].append(
#         [cpoints[i], int(cmaplist[i][1:3], 16) / 256, int(cmaplist[i][1:3], 16) / 256]
#     )
#     cdict["green"].append(
#         [cpoints[i], int(cmaplist[i][3:5], 16) / 256, int(cmaplist[i][3:5], 16) / 256]
#     )
#     cdict["blue"].append(
#         [cpoints[i], int(cmaplist[i][5:7], 16) / 256, int(cmaplist[i][5:7], 16) / 256]
#     )
# autocmap = LinearSegmentedColormap("autocmap", cdict)
# autocolours = {
#     "red1": "#c33248",
#     "blue1": "#84DCCF",
#     "blue2": "#6F8AB7",
#     "redrange": ["#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"],
# }  # '#D95D39'
