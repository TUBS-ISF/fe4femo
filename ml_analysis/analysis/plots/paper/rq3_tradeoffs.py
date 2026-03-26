import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import xscale, xlim

from analysis.analysis_helper import get_modified_performance, get_order, get_modified_task_time, \
    get_replace_dictionary, get_reduction
from analysis.plots.plot_helper import add_median_labels

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file_qual = "model_quality.csv"
file_fsRuntime = "task_times.csv"
file_stability = "sel_stability.csv"
file_reduction = "feature_active.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="Linux Libertine O",)


df_qual = get_modified_performance(path + file_qual)
df_qual = df_qual.set_index(keys=["ml_task", "feature_selector", "ml_model", "fold"])
#df_qual['model_quality'] = df_qual['model_quality'].where(df_qual['model_quality'] >= -1, -1.05)

df_stab = pd.read_csv(path + file_stability, index_col=[0, 1], header=0)
df_stab = df_stab.reset_index().replace(get_replace_dictionary()).drop(columns=["lower", "upper"])#
df_stab = df_stab.set_index(keys=["ml_task", "feature_selector"])
df_stab = pd.concat([df_stab, df_qual.groupby(["ml_task", "feature_selector"]).mean()], axis=1)

df_red = get_reduction(path+file_reduction)
df_red = df_red.set_index(keys=["ml_task", "feature_selector", "ml_model", "fold"])
df_red = pd.concat([df_red, df_qual], axis=1)
df_red['rel_count'] = 1 - df_red['rel_count']

df_fstime = get_modified_task_time(path+file_fsRuntime)
df_fstime = df_fstime.set_index(keys=["ml_task", "feature_selector", "ml_model", "fold"])
df_fstime = pd.concat([df_fstime, df_qual], axis=1)

#TODO: limit, label, marker, maybe mean on folds

fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(15, 3), )

#axs[0].axhline(y=-1.05, c='red', ls='--', lw=1)
plot_fstime = sns.scatterplot(df_fstime.groupby(["feature_selector"]).mean(), x="task_time", y="model_quality", ax=axs[0])
plot_fstime.set(xscale='log', ylim=(-1.005,1.005))
plot_fstime.set(xlabel='', ylabel='Model Quality')

#axs[1].axhline(y=-1.05, c='red', ls='--', lw=1)
plot_red = sns.scatterplot(df_red.groupby(["feature_selector"]).mean(), x="rel_count", y="model_quality", ax=axs[2])
plot_red.set(ylim=(-1.005,1.005), xlim=(-0.005,1.005))
plot_red.set(xlabel='', ylabel='Model Quality')

#axs[2].axhline(y=-1.05, c='red', ls='--', lw=1, label="Cutoff Model Quality")
plot_stab = sns.scatterplot(df_stab.groupby(["feature_selector"]).mean(), x="stability", y="model_quality", ax=axs[1])
plot_stab.set(ylim=(-1.005,1.005), xlim=(-0.005,1.005))
plot_stab.set(xlabel='', ylabel='Model Quality')

#plt.figlegend(loc='upper center', bbox_to_anchor=(0.58, .99),)
#axs[2].get_legend().remove()
fig.tight_layout()

plt.savefig("out/rq3_tradeoffs.pdf")
#plt.show()