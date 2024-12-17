import json
import os
from abc import ABC
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.dataset.metaculus_api import get_all_metaculus_questions
from src.utils import logger


@dataclass
class Question(ABC):
    question_id: str
    title: str
    created_at: datetime
    resolved: bool
    description: str = ""
    news_summary: str = ""

    def __str__(self):
        return self.title


@dataclass
class BinaryQuestion(Question):
    possibilities = [False, True]
    resolution: Optional[bool] = None


class DataJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        if isinstance(obj, (Question, Forecast, EnsembleForecast)):
            data = asdict(obj)
            data["__type__"] = obj.__class__.__name__  # Add type for reconstruction
            return data
        return super().default(obj)


def data_json_decoder(d):
    if "__type__" in d:
        obj_type = d.pop("__type__")  # Remove the __type__ key
        if obj_type == "datetime":
            return datetime.fromisoformat(d["value"])
        if obj_type == "BinaryQuestion":
            return BinaryQuestion(**d)
        if obj_type == "Question":
            return Question(**d)
        if obj_type == "Forecast":
            return Forecast(**d)
        if obj_type == "EnsembleForecast":
            forecasts = [
                Forecast(**f) for f in d["forecasts"]
            ]  # Convert list of dicts to list of Forecasts
            d["forecasts"] = forecasts
            return EnsembleForecast(**d)
    return d


@dataclass
class Forecast(ABC):
    """A forecast made by a single model for a specific question.

    Parameters
    ----------
    prediction : Union[int, float]
        The numerical prediction value.
    reasoning : str
        The explanation or reasoning behind the prediction.
    question_id : Optional[str], optional
        The ID of the question being forecasted.
    model : Optional[str], optional
        The name/version of the model making the forecast.
    prompt_id : Optional[str], optional
        The ID of the prompt used to generate this forecast.
    """

    prediction: Union[int, float]
    reasoning: str
    question_id: Optional[str] = None
    model: Optional[str] = None
    prompt_id: Optional[str] = None


@dataclass
class EnsembleForecast(ABC):
    """A collection of forecasts from multiple models for a specific question.

    Parameters
    ----------
    forecasts : list[Forecast]
        List of individual forecasts from different models.
    question_id : Optional[str], optional
        The ID of the question being forecasted.
    """

    forecasts: list[Forecast]
    question_id: Optional[str] = None

    def prediction(self):
        """Calculate the ensemble prediction using the median of all forecasts.

        Returns
        -------
        float
            The median prediction value across all forecasts in the ensemble.
        """
        predicitons = self._raw_predictions()
        return np.median(predicitons)

    def _raw_predictions(self):
        """Get a list of all individual prediction values.

        Returns
        -------
        list
            List of prediction values from all forecasts in the ensemble.
        """
        return [forecast.prediction for forecast in self.forecasts]


class MetaculusDataset:
    def __init__(
        self, path: str, file_infix: str = None, download_new_data: bool = False
    ):
        self.path = path
        self.file_infix = file_infix
        if download_new_data:
            self.questions = get_all_metaculus_questions()
            self.save()
        else:
            if file_infix:
                files = filter(
                    lambda x: x.startswith(f"metaculus_questions_{file_infix}"),
                    os.listdir(path),
                )
            else:
                files = filter(
                    lambda x: x.startswith("metaculus_questions"), os.listdir(path)
                )
            files = list(files)
            if len(files) == 0:
                raise ValueError(
                    "No dataset found. Set download_new_data=True to download the"
                    " dataset."
                )
            newest_file = max(files)
            self.questions = pd.read_csv(os.path.join(path, newest_file), index_col=0)
        logger.info(f"Questions loaded: {len(self.questions)}")
        time_format = "%Y-%m-%d"
        self.questions["created_at"] = pd.to_datetime(
            self.questions["created_at"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0],
            format=time_format,
        )
        self.questions["actual_resolve_time"] = pd.to_datetime(
            self.questions["actual_resolve_time"].str.extract(r"(\d{4}-\d{2}-\d{2})")[
                0
            ],
            format=time_format,
        )

    def save(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        if not self.file_infix:
            filename = f"metaculus_questions_{current_date}_{self.questions['id'].max()}-{self.questions['id'].min()}.csv"
        else:
            filename = f"metaculus_questions_{self.file_infix}_{current_date}_{self.questions['id'].max()}-{self.questions['id'].min()}.csv"
        self.questions.to_csv(os.path.join(self.path, filename))

    def select_questions_newer_than(self, date: datetime):
        question_count = len(self.questions)
        self.questions = self.questions[self.questions["created_at"] > date]
        logger.info(
            f"Questions newer than {date}: {len(self.questions)} out of"
            f" {question_count}"
        )

    def select_questions_resolved_before(self, date: datetime):
        question_count = len(self.questions)
        self.questions = self.questions[self.questions["actual_resolve_time"] < date]
        logger.info(
            f"Questions resolved before {date}: {len(self.questions)} out of"
            f" {question_count}"
        )

    def select_questions_with_status(self, status: str) -> pd.DataFrame:
        assert status in [
            "upcoming",
            "open",
            "closed",
            "resolved",
        ], f"Invalid status: {status}"
        question_count = len(self.questions)
        self.questions = self.questions[self.questions["status"] == status]
        if status == "resolved":
            self.questions = self.questions[
                (self.questions["resolution"] != "ambiguous")
                & (self.questions["resolution"] != "annulled")
            ]
        logger.info(
            f"Questions with status {status}: {len(self.questions)} out of"
            f" {question_count}"
        )

    def select_questions_with_forecast_type(self, forecast_type: str) -> pd.DataFrame:
        assert forecast_type in [
            "binary",
            "multiple_choice",
            "numeric",
            "date",
        ], f"Invalid question type: {forecast_type}"
        question_count = len(self.questions)
        self.questions = self.questions[self.questions["type"] == forecast_type]
        logger.info(
            f"Questions with type {forecast_type}: {len(self.questions)} out of"
            f" {question_count}"
        )

    def get_question(self, question_id: str) -> Question:
        assert (
            question_id in self.questions.index
        ), f"Question {question_id} not found in dataset."
        question_row = self.questions.loc[self.questions.index == question_id].iloc[0]
        forecast_type = question_row.type
        if forecast_type == "binary":
            if question_row.resolution == "no":
                resolution = False
            elif question_row.resolution == "yes":
                resolution = True
            else:
                resolution = None
            return BinaryQuestion(
                question_id="metaculus-" + str(question_row.id),
                title=question_row.title,
                created_at=question_row.created_at,
                resolved=question_row.status == "resolved",
                resolution=resolution,
                description=question_row.description,
            )
        else:
            raise NotImplementedError(
                f"Forecast type {forecast_type} not implemented yet."
            )
