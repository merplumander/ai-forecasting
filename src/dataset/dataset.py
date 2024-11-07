import datetime
import os
from abc import ABC
from dataclasses import dataclass

import pandas as pd

from src.dataset.metaculus_api import get_all_metaculus_questions


@dataclass
class Question(ABC):
    question_id: str
    title: str
    created_at: datetime.datetime
    resolved: bool
    description: str = ""


@dataclass
class BinaryQuestion(Question):
    possibilities = [False, True]
    resolution: bool


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
        print("Questions loaded:", len(self.questions))
        time_format = "%Y-%m-%d"
        self.questions["created_at"] = pd.to_datetime(
            self.questions["created_at"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0],
            format=time_format,
        )

    def save(self):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if not self.file_infix:
            filename = f"metaculus_questions_{current_date}_{self.questions['id'].max()}-{self.questions['id'].min()}.csv"
        else:
            filename = f"metaculus_questions_{self.file_infix}_{current_date}_{self.questions['id'].max()}-{self.questions['id'].min()}.csv"
        self.questions.to_csv(os.path.join(self.path, filename))

    def select_questions_newer_than(self, date: datetime.datetime):
        question_count = len(self.questions)
        self.questions = self.questions[self.questions["created_at"] > date]
        print(
            f"Questions newer than {date}: {len(self.questions)} out of"
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
                and (self.questions["resolution"] != "annulled")
            ]
        print(
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
        print(
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
