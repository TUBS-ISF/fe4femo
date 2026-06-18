import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_order, get_modified_task_time, load_multiindex
from analysis.plots.plot_helper import add_median_labels, draw_shades

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "model_times.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="FreeSerif",)


df = load_multiindex(path+file)
df = df.replace({"Random Forest":"R. Forest"})
translator_dict = {'model_eval':'Prediction', 'model_training':'Training'}
df = df.rename(columns=translator_dict)
print(df)
df = df.reset_index().melt(id_vars=['feature_selector', 'ml_model', 'ml_task', 'fold'], var_name="task", value_name="time", value_vars=translator_dict.values())
print(df)

plot = sns.catplot(df, x="time", y="ml_model",  hue="task", hue_order=[translator_dict['model_training'], translator_dict['model_eval']], estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", line_kws={"color":"c", "linewidth":2}, legend_out=False)
plot.set(ylabel="ML Model", xscale='log', xlabel="Runtime [s]")
plot.set_titles(col_template="", )
sns.despine(left=True, bottom=True)

#for ax in plot.axes.flat:
#    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)

plot.figure.set_size_inches(2*3.48, 2*6*0.2)
sns.move_legend(plot, "upper right", title="")

plot.tight_layout()

draw_shades(plot.figure, plot.figure.axes[0], plot.figure.axes[0])

#plt.show()
plot.savefig("out/rq1_model_training_eval.pdf")