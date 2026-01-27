import numpy as np
from collections import namedtuple

scale = {
    "figure": 10.8,
    "title": 20,
    "label": 22,
    "legend": 16,
    "ticks": 20
}
title = "Phase-2 Simulation Work in progress"
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
Thresholds = namedtuple('Thresholds', ['l1_pt', 'l1_mass', 'gen_pt', 'gen_mass'])
iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))