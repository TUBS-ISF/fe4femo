from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from seaborn._core.groupby import GroupBy
from seaborn._core.moves import Move
from seaborn._core.typing import Default
from seaborn._core.scales import Scale
from functools import partial

from analysis.analysis_helper import get_order, get_replace_dictionary

@dataclass
class SemiStack(Move):
    """
    Displacement of overlapping bar or area marks along the value axis.
    Examples
    --------
    .. include:: ../docstrings/objects.Stack.rst
    """

    def _stack(self, df, orient, order=None):

        # TODO should stack do something with ymin/ymax style marks?
        # Should there be an upstream conversion to baseline/height parameterization?
        df = GroupBy(order).apply(df, lambda x: x)
        if df["baseline"].nunique() > 1:
            err = "Stack move cannot be used when baselines are already heterogeneous"
            raise RuntimeError(err)
        other = {"x": "y", "y": "x"}[orient]
        stacked_lengths = (df[other] - df["baseline"]).dropna().cumsum()
        offsets = stacked_lengths.shift(1).fillna(0)
        df[other] = stacked_lengths
        df["baseline"] = df["baseline"] + offsets
        return df
    def __call__(
        self, data: pd.DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> pd.DataFrame:
        # TODO where to ensure that other semantic variables are sorted properly?
        # TODO why are we not using the passed in groupby here?
        groupers = ["col", "row", orient]
        return GroupBy(groupers).apply(data,
                                       partial(self._stack, order=groupby.order),
                                       orient)

sns.set_theme(style="whitegrid", palette="colorblind")

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_active.csv"
path_groups = "/home/ubuntu/MA/raphael-dunkel-master/data/featureExtraction/groupMapping.csv"

groups = pd.read_csv(path_groups, index_col=0, header=0)['groupName']
group_count = groups.value_counts().to_dict()
groups = groups.to_dict()

df = pd.read_csv(path, index_col=[0, 1, 2, 3, 4, 5, 6])
df = df[df.index.get_level_values(5) == False]

df.reset_index(inplace=True)
selector_count = df['feature_selector'].nunique()

df.replace({False: 0, True: 1}, inplace=True)
#df.drop(columns=["ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], inplace=True)

df = df.groupby(["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"]).sum()

df.reset_index(inplace=True)
df = df.melt(id_vars=["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], var_name="feature", value_name="count")


#transform to groups
df['group'] = df['feature'].apply(lambda x: groups[x])
df.drop(columns=["feature"], inplace=True)
df = df.groupby(["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold", 'group']).any().reset_index()
df.replace({False: 0, True: 1}, inplace=True)
df['max_count'] = 1
df.drop(columns=["ml_task", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], inplace=True)
print(df)
df = df.groupby(['group', 'feature_selector']).sum().reset_index()

print(df)
df['rel_count'] =  df['count'] / df['max_count']
df['rel_count'] = df['rel_count'] / selector_count
print(df)
df.replace(get_replace_dictionary(), inplace=True)
print(df)

map_dict = {
    "FMBA" : "FMBA",
    "SATZilla2024": "SATZilla",
    "SATfeatPy":"SATfeatPy",
    "FM":"FM Fact Label",
    "DyMMer":"DyMMer",
    "ESYES":"ESYES"
}

def groupTransformer(x : str) -> str:
    no = x.split("_")[-1]
    name = x.split("_")[0]
    return f"{map_dict[name]} {no}"

df['group'] = df['group'].apply(groupTransformer)
order = df.groupby(['group', 'feature_selector'])['rel_count'].mean().groupby(['group']).sum().sort_values(ascending=False).index.values

plot = (
    so.Plot(df.rename(columns={"feature_selector":"Feature Selector"}), y="group", x="rel_count", color="Feature Selector", )
    #.facet(row="ml_task")
    .add(so.Bar(), so.Agg("mean"), SemiStack())
    #.add(so.Text(color="k"), MiddleStack())
    .layout(size=(10,14))
    .scale(y=so.Nominal(order=order))
    .label(legend="Feature Selector",x="Percentage of ML Pipeline Trainings With Group Active", y="Feature Group")
    .scale(color=so.Nominal(order=get_order()))
)
plot.save("out/rq3_feature_group_count.pdf", bbox_inches='tight')
#plot.show()