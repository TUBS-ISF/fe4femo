import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from analysis.analysis_helper import get_modified_performance, get_order
from analysis.plots.plot_helper import add_median_labels, draw_shades

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "model_quality.csv"

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.4, font="FreeSerif",)


df = get_modified_performance(path+file)

df_fsel = df.copy().reset_index()
df_fsel['mod'] = 'Feature Selector'
df_fsel.rename(columns={'feature_selector':'y'}, inplace=True)
df_fsel['feature_selector'] = 'a'
df_fsel.sort_values(by="y", key=lambda column: column.map(lambda e: get_order().index(e)), inplace=True)

df_mlmodel = df.copy().reset_index()
df_mlmodel['mod'] = 'ML Model'
df_mlmodel.rename(columns={'ml_model':'y'}, inplace=True)
df_mlmodel['ML Model'] = 'a'


df_data = pd.concat([df_fsel, df_mlmodel], ignore_index=True)


plot = sns.catplot(df_data, x="model_quality", y="y", col="ml_task", col_order=['Runtime Kissat', 'Runtime CaDiBack', 'Runtime Spur', 'CM Cardinality', 'Backbone Size', '#SAT Alg. Sel.'], estimator="median", errorbar="sd", kind="box", orient="h", facet_kws={"xlim":(-1,1)}, legend="auto", medianprops={"linewidth": 2, "color":"c"}, sharex=True, sharey=True)
plot.refline(x=0, color="r", linestyle="--")
plot.set(xlim=(-1,1), xlabel="")
plot.set_titles(col_template="{col_name}", row_template="{row_name}")
plot.set(ylabel=" ")

#TODO Beschriftungen für y-Achse
#TODO Farblich hinterlegen?


plot.figure.set_size_inches(2*7.13, (16+6)*0.2)



sns.despine(left=True, bottom=True)
plot.tight_layout()

for ax in plot.figure.axes:
    ax.xaxis.set_major_locator(ticker.LinearLocator(5))

plot.figure.text(0.016,0.465,"Feature Selector", fontsize="large", color="purple", rotation="vertical", horizontalalignment="center")
plot.figure.text(0.016,0.125,"ML Model", fontsize="large", color="green", rotation="vertical", horizontalalignment="center")



start, width, height = draw_shades(plot.figure, plot.axes[0,0], plot.axes[0,5])

trans = blended_transform_factory(plot.figure.transFigure, plot.figure.axes[0].transData)
r = Rectangle(xy=(start,0-0.5), width=width, height=height*16, transform=trans, color='plum', zorder=0.1, fill=True, lw=0, alpha=0.1)
plot.figure.add_artist(r)
r = Rectangle(xy=(start,16-0.5), width=width, height=height*6, transform=trans, color='palegreen', zorder=0.1, fill=True, lw=0, alpha=0.1)
plot.figure.add_artist(r)

plot.savefig("out/rq2_model_qual.pdf")
#plt.show()