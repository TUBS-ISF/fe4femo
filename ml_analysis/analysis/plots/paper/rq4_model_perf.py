import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_replace_dictionary


def plot_dots(df_in, goal, marker, color, label):
    xmarker = []
    ymarker = []
    modifier = [-0.36, -0.28, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.28, 0.36]
    df_mod = df_in.groupby(["ml_task"])[goal].first()

    for ymod, task in enumerate(
            ['Runtime Kissat', 'Runtime CaDiBack', 'Runtime Spur', 'CM Cardinality', 'Backbone Size',
             '#SAT Algorithm Selection']):
            ymarker.append(ymod)
            xmarker.append(
                df_mod[task]
            )

    ax.scatter(xmarker, ymarker, marker=marker, s=25, label=label, color=color, zorder=4)

path = "/mnt/c/Users/rsd61/IdeaProjects/ma-raphael-dunkel/data/extracted_ml_results/"
file = "model_quality_enriched.csv"

plt.style.use("seaborn-v0_8-whitegrid")
plt.style.use("seaborn-v0_8-colorblind")
#sns.set(context="paper", style="whitegrid", palette="colorblind", font_scale=1.3, font="Linux Libertine O")
sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.1, font="Linux Libertine O", rc={'ytick.labelsize': 8})

df = pd.read_csv(path+file, index_col=[0,1,2,3]).reset_index()
df.replace({"FM Cardinality":"CM Cardinality"}, inplace=True)
df['fold'] = df['fold'].map(lambda x: f"Fold {x}")
df.set_index(keys=["ml_task", "feature_selector", "ml_model", "fold"], inplace=True)
df = df.groupby(["ml_task", "feature_selector", "ml_model"]).mean()


#fig, ax = plt.subplots(figsize=(12,7))

#palette = sns.color_palette([(0.00392156862745098, 0.45098039215686275, 0.6980392156862745)], n_colors=10)
#palette = sns.color_palette([(0.33725490196078434, 0.7058823529411765, 0.9137254901960784)], n_colors=10)

ax = sns.boxplot(df, x="model_quality", y="ml_task", order=['Runtime Kissat', 'Runtime CaDiBack', 'Runtime Spur', 'CM Cardinality', 'Backbone Size', '#SAT Algorithm Selection'], orient="h", medianprops={"linewidth": 1.5, }, fliersize=1 )
ax.set_xlim(-1,1.005)
ax.set_xlabel("Model Quality")
ax.set_ylabel("ML Task")

plot_dots(df, "SBM", 'd', 'green', "Single Best Configuration")
plot_dots(df, "VBM", 'x', 'orange', "Virtual Best Configuration")

plt.gcf().set_size_inches(6,3)

plt.legend(bbox_to_anchor=(0.48, 1), loc='lower center', borderaxespad=0., ncol=2, fancybox=True, title="")

sns.despine(left=True, bottom=True)
plt.tight_layout()


plt.savefig("out/rq4_model_qual_model.pdf")
#plt.show()