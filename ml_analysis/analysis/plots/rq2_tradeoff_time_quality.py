from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.analysis_helper import get_modified_performance, get_modified_feature_time, get_modified_task_time, \
    get_order

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/"

sns.set_theme(style="whitegrid", palette="colorblind")

series = [
    get_modified_performance(path+"model_quality.csv").set_index(['feature_selector', 'ml_model', 'ml_task', 'fold'])['model_quality'],
    get_modified_feature_time(path+"feature_times.csv").groupby(['feature_selector', 'ml_model', 'ml_task', 'fold']).median()['feature_time'],
    get_modified_task_time(path+"task_times.csv").set_index(['feature_selector', 'ml_model', 'ml_task', 'fold'])['task_time']
]

df = pd.concat(series, axis=1)

print(df)
df = df.groupby(level=['feature_selector', 'ml_task']).median()
df = df.reset_index()
translator_dict = {'feature_time':"Cumulated Computation Time of Selected Features [s]", 'task_time':"Runtime of Singular Feature Selector Execution [s]"}
df = df.rename(columns=translator_dict)
print(df)

plot = (so.Plot(df, y="model_quality", color="feature_selector", marker="feature_selector")
        .pair(x=[translator_dict['feature_time'], translator_dict["task_time"]]).facet(row="ml_task", order=['Runtime Kissat', 'Runtime CaDiBack', 'Runtime Spur', 'FM Cardinality', 'Backbone Size', '#SAT Algorithm Selection'])
        .add(so.Dot()).limit(y=(-1,1))
        .layout(size=(15,20)).scale(x="log", color=so.Nominal(order=get_order()))
        .label(legend="Feature Selector", y="Model Quality"))
plt.tight_layout()
#plot.show()
plot.save("out/rq2_tradeoff_qual_time.pdf", bbox_inches='tight')
