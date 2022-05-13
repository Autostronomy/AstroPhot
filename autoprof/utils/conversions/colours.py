import numpy as np
from matplotlib.colors import LinearSegmentedColormap


cmaplist = ["#000000", "#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"]
cdict = {"red": [], "green": [], "blue": []}
cpoints = np.linspace(0, 1, len(cmaplist))
for i in range(len(cmaplist)):
    cdict["red"].append(
        [cpoints[i], int(cmaplist[i][1:3], 16) / 256, int(cmaplist[i][1:3], 16) / 256]
    )
    cdict["green"].append(
        [cpoints[i], int(cmaplist[i][3:5], 16) / 256, int(cmaplist[i][3:5], 16) / 256]
    )
    cdict["blue"].append(
        [cpoints[i], int(cmaplist[i][5:7], 16) / 256, int(cmaplist[i][5:7], 16) / 256]
    )
autocmap = LinearSegmentedColormap("autocmap", cdict)
autocolours = {
    "red1": "#c33248",
    "blue1": "#84DCCF",
    "blue2": "#6F8AB7",
    "redrange": ["#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"],
}  # '#D95D39'

