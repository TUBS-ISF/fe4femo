import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_order, get_modified_feature_time
from analysis.plots.plot_helper import add_median_labels

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "feature_times.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="Linux Libertine O", rc={'xtick.labelsize': 14, 'ytick.labelsize': 11})

df = get_modified_feature_time(path+file)

plot = sns.catplot(df,  x="feature_time", y="feature_selector", estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", height=5, aspect=1.5, line_kws={"linewidth": 2},  order=get_order())
plot.set(xlabel="Cumulated Computation Time of Selected Features [s]", xscale='log')
sns.despine(left=True, bottom=True)

#for ax in plot.axes.flat:
#    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)

plot.tight_layout()

#plt.show()
plot.savefig("out/rq12_feature_set_time.pdf")