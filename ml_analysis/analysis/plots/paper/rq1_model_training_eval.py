import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_order, get_modified_task_time, load_multiindex
from analysis.plots.plot_helper import add_median_labels

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "model_times.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.2, font="Linux Libertine O", rc={'xtick.labelsize': 12, 'ytick.labelsize': 9})


df = load_multiindex(path+file)
translator_dict = {'model_eval':'Final ML Model Prediction', 'model_training':'Final ML Model Training'}
df = df.rename(columns=translator_dict)
print(df)
df = df.reset_index().melt(id_vars=['feature_selector', 'ml_model', 'ml_task', 'fold'], var_name="task", value_name="time", value_vars=translator_dict.values())
print(df)

plot = sns.catplot(df, x="time", y="ml_model",  col="task", col_order=[translator_dict['model_training'], translator_dict['model_eval']], estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", height=2.5, aspect=2, line_kws={"linewidth": 2})
plot.set(ylabel="ML Model", xscale='log', xlabel="")
plot.set_titles(col_template="")
sns.despine(left=True, bottom=True)

#for ax in plot.axes.flat:
#    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)



plot.tight_layout()

#plt.show()
plot.savefig("out/rq1_model_training_eval.pdf")