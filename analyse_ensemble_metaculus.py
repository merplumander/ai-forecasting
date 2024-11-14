# %%
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.dataset.dataset import MetaculusDataset
from src.forecasting.utils import brier_score, load_raw_forecasting_data

ROOT = Path(__file__).parent
# %%
forecasts = load_raw_forecasting_data(ROOT / "forecasts")
dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_08_01_to_election_",
    download_new_data=False,
)
# %%
# count datapoints
forecasts.groupby("model_name").count()
# %%
ax = forecasts.boxplot(column="forecast", by="model_name", vert=False)
plt.show()
# %%
aggregated_forecasts = forecasts.groupby(["model_name", "question_id"])["forecast"].agg(
    ["median", "mean", "std"]
)
# %%
std_matrix = aggregated_forecasts.reset_index().pivot(
    index="model_name", columns="question_id", values="std"
)
sns.heatmap(std_matrix, xticklabels=False, cbar_kws={"label": "SD of forecasts"})
# %%
mean_matrix = aggregated_forecasts.reset_index().pivot(
    index="model_name", columns="question_id", values="mean"
)
sns.heatmap(mean_matrix, xticklabels=False, cbar_kws={"label": "Mean of forecasts"})
# %%
median_matrix = aggregated_forecasts.reset_index().pivot(
    index="model_name", columns="question_id", values="median"
)
sns.heatmap(median_matrix, xticklabels=False, cbar_kws={"label": "Median of forecasts"})
# %%
median_matrix = aggregated_forecasts.reset_index().pivot(
    index="question_id", columns="model_name", values="median"
)
corr_matrix = median_matrix.corr()
sns.heatmap(corr_matrix, cbar_kws={"label": "Correlation"}, vmin=0.4, vmax=1)
# %%
mean_matrix = aggregated_forecasts.reset_index().pivot(
    index="question_id", columns="model_name", values="mean"
)
corr_matrix = mean_matrix.corr()
sns.heatmap(corr_matrix, cbar_kws={"label": "Correlation"}, vmin=0.4, vmax=1)
# %%
resolutions = []
ensemble_medians = []
metaculus_predictions = []
for question_id in median_matrix.index:
    resolution = dataset.get_question(question_id).resolution
    resolutions.append(resolution)
    ensemble_median = forecasts[forecasts["question_id"] == question_id][
        "forecast"
    ].median()
    ensemble_medians.append(ensemble_median)
    latest = metaculus_prediction = eval(
        dataset.questions.loc[question_id].aggregations
    )["metaculus_prediction"]["latest"]
    if latest is not None:
        metaculus_predictions.append(latest["forecast_values"][1])
    else:
        metaculus_predictions.append(np.nan)


brier_scores = median_matrix.apply(
    lambda column: brier_score(column / 100, resolutions), axis=0
)
brier_scores
# %%
# brier score if we take median of ensemble as forecast
brier_score(np.array(ensemble_medians) / 100, resolutions)
# %%
brier_score(np.array(metaculus_predictions) / 100, resolutions)

# %%
