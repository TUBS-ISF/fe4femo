import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_order
from analysis.plots.plot_helper import add_median_labels

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "model_quality.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="Linux Libertine O", rc={'xtick.labelsize': 14, 'ytick.labelsize': 11})


df = get_modified_performance(path+file)

plot = sns.catplot(df, x="model_quality", y="feature_selector", col="ml_task", col_order=['Runtime Kissat', 'Runtime CaDiBack', 'Runtime Spur', 'CM Cardinality', 'Backbone Size', '#SAT Algorithm Selection'], estimator="median", order=get_order(), errorbar="sd", kind="box", orient="h", facet_kws={"xlim":(-1,1)}, legend="auto", medianprops={"linewidth": 2}, height=4, aspect=0.8)
plot.refline(x=0, color="r", linestyle="--")
plot.set(xlim=(-1,1), ylabel="Feature Selector", xlabel="")
plot.set_titles(col_template="{col_name}")

sns.despine(left=True, bottom=True)
plot.tight_layout()

plot.savefig("out/rq21_model_qual_selector.pdf")
#plt.show()