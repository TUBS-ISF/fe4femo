import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_order, get_modified_task_time, get_modified_feature_time
from analysis.plots.plot_helper import add_median_labels

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.2, font="Linux Libertine O", rc={'xtick.labelsize': 12, 'ytick.labelsize': 10})

print("Test")
df_task = get_modified_task_time(path+"task_times.csv").rename(columns={"task_time": "time"})
df_sel = get_modified_feature_time(path+"feature_times.csv").rename(columns={"feature_time": "time"})
#df_sel = df_sel.groupby(['feature_selector', 'ml_model', 'ml_task', 'fold'])['time'].median().reset_index()

df_sel['dif'] = "Cumulated Computation Time of Selected Features"
df_task['dif'] = "Singular Feature Selector Execution"

df = pd.concat([df_task, df_sel], sort=False)
print(df)


plot = sns.catplot(df, x="time", y="feature_selector", col="dif", estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", height=3.5, aspect=2, line_kws={"linewidth": 2},  order=get_order())
plot.set(xlabel="", ylabel="Feature Selector", xscale='log')
plot.set_titles(col_template="")
sns.despine(left=True, bottom=True)

#for ax in plot.axes.flat:
#    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)


plot.tight_layout()

#plt.show()
plot.savefig("out/rq1_feature_times.pdf")