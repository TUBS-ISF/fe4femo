import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import scipy.stats as stats

import math

from analysis.analysis_helper import get_replace_dictionary, get_order, get_reduction

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.1, font="Linux Libertine O", rc={'xtick.labelsize': 11, 'ytick.labelsize': 8})

df_stab = pd.read_csv(path + "sel_stability.csv", index_col=[0, 1], header=0)
df_red = get_reduction(path+"feature_active.csv")
df_red['rel_count'] = 1 - df_red['rel_count']

df_stab = df_stab.reset_index()
df_stab.replace(get_replace_dictionary(), inplace=True)
df_stab.drop(columns=["lower", "upper"])
print(df_stab)

df_stab = df_stab.groupby("feature_selector")[['stability', 'variance']].mean()
alpha = 0.05
df_stab['lower'] = df_stab['stability'] - stats.norm.ppf(1 - alpha / 2) * np.sqrt(df_stab['variance'])  # lower bound of the confidence interval at a level alpha
df_stab['upper'] = df_stab['stability'] + stats.norm.ppf(1 - alpha / 2) * np.sqrt(df_stab['variance'])  # upper bound of the confidence interval




fig, axs = plt.subplots(ncols=2, sharey=True)
plot1 = sns.boxenplot(ax=axs[1], x="rel_count", y="feature_selector", data=df_red, order=get_order())
plot1.set(xlim=(0,1.005), ylabel="Feature Selector", xlabel="")


plot = so.Plot(df_stab, y="feature_selector", x="stability", xmin="lower", xmax="upper").add(so.Bar()).add(so.Range(linewidth=2)).scale(y=so.Nominal(order=get_order()))
#plot = plot.layout(size=(8,12))
plot = plot.limit(xlim=(0,1))
plot = plot.label(x="", y="Feature Selector")
plot.on(axs[0]).plot()
axs[0].set_xlim(0,1.005)

fig.set_figheight(2.5)
fig.set_figwidth(12)
sns.despine(left=True, bottom=True)

plt.subplots_adjust(wspace=5.0)
plt.tight_layout()
#plt.show()
plt.savefig("out/rq21_stability_reduction.pdf")

