import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_forecasting_data(path: Path, contain_prompt_ids=False) -> pd.DataFrame:
    """Load raw forecasting data from a directory where each file contains the
    forecasts for a single question.

    Parameters
    ----------
    path : Path
        Path to the directory containing the forecasting data.
    contain_prompt_ids : bool, optional
        Whether the data contains prompt_ids, by default False

    Returns
    -------
    pd.DataFrame:
        DataFrame containing data.
    """
    data = pd.DataFrame(
        {"question_id": [], "model_name": [], "forecast": [], "explanation": []}
    )
    if contain_prompt_ids:
        data = pd.DataFrame(
            {
                "question_id": [],
                "model_name": [],
                "prompt_ids": [],
                "forecast": [],
                "explanation": [],
            }
        )
    l = os.listdir(path)
    li = [x.split(".")[0] for x in l]
    for question_id in li:
        with open(path / f"{question_id}.txt", "r") as file:
            lines = file.readlines()
        tuples_list = [eval(line.strip()) for line in lines]
        data = pd.concat(
            [data, pd.DataFrame(tuples_list, columns=data.columns)], ignore_index=True
        )
    return data


def brier_score(forecasts, targets):
    """Calculate the Brier score for a set of forecasts and targets.

    Parameters
    ----------
    forecasts : np.array
        Array of forecasted probabilities.
    targets : np.array
        Array of binary targets.

    Returns
    -------
    float:
        Brier score.
    """
    return np.nanmean((forecasts - targets) ** 2)
