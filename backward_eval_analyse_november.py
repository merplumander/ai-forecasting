# %%
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.analysis.utils import brier_score, load_raw_forecasting_data
from src.dataset.dataset import MetaculusDataset
from src.utils import ROOT

# %%
forecasts = load_raw_forecasting_data(ROOT / "forecasts" / "backward-evaluation")
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
max_days = 100
metaculus_predictions_n_days_after_start = np.empty(
    (len(median_matrix.index), max_days)
)
for i, question_id in enumerate(median_matrix.index):
    resolution = dataset.get_question(question_id).resolution
    resolutions.append(resolution)
    ensemble_median = forecasts[forecasts["question_id"] == question_id][
        "forecast"
    ].median()
    metaculus_prediction_history = eval(
        dataset.questions.loc[question_id].aggregations
    )["recency_weighted"]["history"]
    if len(metaculus_prediction_history) > 0:
        metaculus_prediction_df = pd.DataFrame(metaculus_prediction_history)
        metaculus_prediction_df["start_time"] = pd.to_datetime(
            metaculus_prediction_df["start_time"], unit="s"
        )
        metaculus_prediction_df["end_time"] = pd.to_datetime(
            metaculus_prediction_df["end_time"], unit="s"
        )
        question_created_at = dataset.questions.loc[question_id].created_at
        for j in range(0, max_days):
            preds = metaculus_prediction_df[
                (metaculus_prediction_df["end_time"] - question_created_at).abs()
                <= pd.Timedelta(days=j + 1)
            ]
            if len(preds) > 0:
                latest = preds.iloc[-1]
            else:
                latest = metaculus_prediction_df.iloc[3]
            if latest["centers"] is not None:
                metaculus_predictions_n_days_after_start[i, j] = latest["centers"][0]
            else:
                metaculus_predictions_n_days_after_start[i, j] = np.nan
    else:
        metaculus_predictions_n_days_after_start[i] = np.empty(max_days)


brier_scores = median_matrix.apply(
    lambda column: brier_score(column / 100, resolutions), axis=0
)
brier_scores
# %%
# brier score if we take median of ensemble as forecast
brier_score(np.array(ensemble_medians) / 100, resolutions)
# %%
brier_scores_days = []
for j in range(max_days):
    brier_scores_days.append(
        brier_score(metaculus_predictions_n_days_after_start[:, j], resolutions)
    )
plt.plot(range(1, max_days + 1), brier_scores_days)
plt.xlabel("Days after question start")
plt.ylabel("Brier score")
plt.legend(["Metaculus recency_weighted"])

# %%
