import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_order
from analysis.plots.plot_helper import add_median_labels


path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/model_quality.csv"

sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path)
df.set_index(["fold", "ml_task", "feature_selector", "ml_model"], inplace=True)
print(df)

idx = pd.IndexSlice


selector, model = df.groupby(by=["feature_selector", "ml_model"])['model_quality'].median().idxmax()
df_sbm = df.loc[idx[:, :, selector, model],'model_quality']
print(df_sbm)
df['SBM'] = df_sbm.reset_index().set_index(["fold","ml_task"])['model_quality']

df_vbm = df.groupby(by=["ml_task", "feature_selector", "ml_model"])['model_quality'].median().groupby("ml_task").idxmax()
df_modded = [ df.loc[idx[:, task, selector, model],'model_quality'] for task, selector, model in df_vbm.to_list()]
df_vbm = pd.concat(df_modded)
print(df_vbm)
df['WBM'] = df_vbm.reset_index().set_index(["fold","ml_task"])['model_quality']


df_idmax = df.loc[df.groupby(by=["fold","ml_task"])['model_quality'].idxmax()].reset_index().set_index(["fold","ml_task"])
df['idmax'] = df_idmax['model_quality']
print(df_idmax)

print(df)


# plot = sns.barplot(df, y="ml_task", x="diff", hue="fold", orient="h", estimator="median")
# #plot.set(ylabel="Feature Selector", xlabel="Model Quality")
# #plot.legend.set_title("ML Model")
#
# plt.tight_layout()
# plt.show()
# #plot.savefig(f"out/rq1_box_{rev_map[name]}.pdf")
