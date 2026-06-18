import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


# https://stackoverflow.com/a/63295846
def add_median_labels(ax: plt.Axes, fmt:int = 2, size='x-small', boxen=False, scientific=False) -> None:
    fmt = f".{fmt}{"E" if scientific else "f"}"
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 0 if boxen else 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       color='white', size=size)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=median.get_color()),
            path_effects.Normal(),
        ])

from itertools import cycle

def draw_shades(fig, ax_start, ax_end, **kwargs):
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    plt.gcf().canvas.draw()

    yticks = ax_start.get_yticks()

    trans = blended_transform_factory(fig.transFigure, ax_start.transData)

    end, _  = fig.transFigure.inverted().transform(ax_end.transAxes.transform([1, 0]))

    height=1

    y_label = ax_start.yaxis.get_label()
    start, _ = fig.transFigure.inverted().transform(y_label.get_transform().transform(y_label.get_position()))
    sol_start = start
    sol_end, _ = fig.transFigure.inverted().transform(ax_start.transAxes.transform([0, 0]))

    for ytick, isShade in zip(yticks, cycle([False, True])):
        if isShade:
            r = Rectangle(xy=(start,ytick-0.5), width=end-start, height=height, transform=trans, color='lightgrey', zorder=0, fill=True, lw=0, alpha=0.5)
            fig.add_artist(r)

    for i in range(1, len(fig.axes)):
        ax_before = fig.axes[i-1]
        ax_after = fig.axes[i]

        start, _ = fig.transFigure.inverted().transform(ax_before.transAxes.transform([1,0]))
        end, _ = fig.transFigure.inverted().transform(ax_after.transAxes.transform([0,0]))

        pos_x = start + (end - start) / 2

        _, y1 = fig.transFigure.inverted().transform(ax_before.transAxes.transform([0, 0]))
        _, y2 = fig.transFigure.inverted().transform(ax_before.transAxes.transform([0, 1]))

        fig.add_artist(Line2D((pos_x, pos_x), (y1,y2), transform=fig.transFigure, color='grey', linewidth=2))

        return sol_start, sol_end-sol_start, height

