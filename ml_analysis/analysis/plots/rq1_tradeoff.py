import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_modified_feature_time, get_modified_task_time, \
    get_replace_dictionary, get_order, get_reduction
from analysis.plots.plot_helper import add_median_labels

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/"

sns.set_theme(style="whitegrid", palette="colorblind")


df_qual = pd.read_csv(path+"model_quality.csv", index_col=[0, 1, 2, 3, 4, 5, 6])['model_quality']
df_qual = df_qual.groupby(['feature_selector', 'ml_task']).median().reset_index()
df_qual = df_qual.replace(get_replace_dictionary())
df_qual.set_index(['feature_selector', 'ml_task'], inplace=True)

df_stab = pd.read_csv(path+"sel_stability.csv", index_col=[0,1], header=0)['stability']
df_stab = df_stab.reset_index().replace(get_replace_dictionary())
df_stab.set_index(['feature_selector', 'ml_task'], inplace=True)

df_red = get_reduction(path+"feature_active.csv")
df_red = df_red.groupby(['feature_selector', 'ml_task'])['rel_count'].median()

df = pd.concat([df_qual, df_stab, df_red], axis=1).reset_index()
df.rename(columns={"feature_selector":"Feature Selector"}, inplace=True)
df.replace(get_replace_dictionary(), inplace=True)
print(df)

plot = sns.relplot(df, x="stability", y="model_quality", hue="Feature Selector", hue_order=get_order(), style="Feature Selector", col="ml_task", col_wrap=2, kind="scatter", facet_kws={"legend_out": True}, height=4)
plot.refline(y=0, color="r", linestyle="--")
plot.set_titles(col_template="{col_name}")
plot.set(ylim=(-1,1.02), xlim=(0,1.05), xlabel="Selection Stability", ylabel="Model Quality", )
plot.tight_layout()

plot.savefig("out/rq1_tradeoff_qual_stability.pdf")
#plt.show()

plt.figure()
plot = sns.relplot(df, x="rel_count", y="model_quality", hue="Feature Selector", hue_order=get_order(), style="Feature Selector", col="ml_task", col_wrap=2, kind="scatter", facet_kws={"legend_out": True}, height=4)
plot.refline(y=0, color="r", linestyle="--")
plot.set_titles(col_template="{col_name}")
plot.set(ylim=(-1,1.02), xlim=(0,1.05), xlabel="Relative Size of Selected Feature Subset", ylabel="Model Quality", )
plot.tight_layout()

plot.savefig("out/rq1_tradeoff_qual_reduction.pdf")
#plt.show()