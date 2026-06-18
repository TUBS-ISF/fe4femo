import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_order, get_modified_task_time, get_modified_feature_time
from analysis.plots.plot_helper import add_median_labels, draw_shades

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="FreeSerif",)



df_task = get_modified_task_time(path+"task_times.csv").rename(columns={"task_time": "time"})
df_sel = get_modified_feature_time(path+"feature_times.csv").rename(columns={"feature_time": "time"})
#df_sel = df_sel.groupby(['feature_selector', 'ml_model', 'ml_task', 'fold'])['time'].median().reset_index()

df_sel['dif'] = "F. Computation"
df_task['dif'] = "F. Selector"

df = pd.concat([df_task, df_sel], sort=False)
print(df)


plot = sns.catplot(df, x="time", y="feature_selector", hue="dif", estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", line_kws={"color":"c", "linewidth":2},  order=get_order(), legend_out=False)
plot.set(xlabel="Runtime [s]", ylabel="Feature Selector", xscale='log')
plot.set_titles(col_template="")
sns.despine(left=True, bottom=True)

#for ax in plot.axes.flat:
#    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)

plot.figure.set_size_inches(2*3.48, 1.4*16*0.2)
sns.move_legend(plot, "upper right", title="", fontsize="small", labelspacing=0.3, borderpad=0.3, handletextpad=0.5)


plot.tight_layout()

draw_shades(plot.figure, plot.figure.axes[0], plot.figure.axes[0])

#plt.show()
plot.savefig("out/rq1_feature_times.pdf")