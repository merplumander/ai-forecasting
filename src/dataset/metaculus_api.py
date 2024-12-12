# %%
import json
from typing import List

import pandas as pd
import requests

from src.utils import logger

API_BASE_URL = "https://www.metaculus.com/api2"


def get_posts_with_offset_and_limit(offset=0, limit=1000) -> List:
    """Download posts from Metaculus with a given offset and limit.

    Parameters
    ----------
    offset : int, optional
        Start, by default 0
    limit : int, optional
        End, by default 1000

    Returns
    -------
    list
        List of questions.
    """
    # get all question types except conditionals
    url = f"https://www.metaculus.com/api/posts/?forecast_type=numeric%2Cdate%2Cbinary%2Cmultiple_choice%2Cgroup_of_questions&order_by=-published_at&limit={limit}&offset={offset}&with_cp=false"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["results"]
    else:
        logger.error(f"Scraping from metaculus failed with code {response.status_code}")
        return None


def get_all_metaculus_questions() -> pd.DataFrame:
    """Download all questions from Metaculus and return them a DataFrame.


    Returns
    -------
    pd.DataFrame
        DataFrame containing all questions from Metaculus.
    """
    questions = []
    offset = 0
    limit = 1000
    while True:
        logger.info(f"Downloading posts: {offset} to {limit} from metaculus.")
        batch = get_posts_with_offset_and_limit(offset=offset, limit=limit)
        if not batch:  # Stop if no more results are returned
            break
        questions.extend(batch)
        offset = limit  # Move to the next set of questions
        limit += 1000  # Increase the limit to get more questions
    posts_with_groups_of_questions = pd.DataFrame(questions).dropna(
        subset=["group_of_questions"]
    )
    questions = pd.DataFrame(questions).dropna(subset=["question"])
    extracted_questions = []
    for _, post in posts_with_groups_of_questions.iterrows():
        for question in post["group_of_questions"]["questions"]:
            if question["description"] == "":
                question["description"] = post["group_of_questions"]["description"]
            if question["resolution_criteria"] == "":
                question["resolution_criteria"] = post["group_of_questions"][
                    "resolution_criteria"
                ]
            extracted_questions.append(question)
    questions_df = pd.DataFrame(questions.question.to_list() + extracted_questions)
    questions_df["question_id"] = "metaculus-" + questions_df["id"].astype(str)
    questions_df.set_index("question_id", inplace=True)
    logger.info(f"Total questions downloaded: {len(questions_df)}.")
    return pd.DataFrame(questions_df)


def post_question_comment(
    question_id: int, comment_text: str, metaculus_token: str
) -> None:
    """
    Post a comment on the question page as the bot user.
    """
    auth_headers = {"headers": {"Authorization": f"Token {metaculus_token}"}}
    response = requests.post(
        f"{API_BASE_URL}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        **auth_headers,
    )
    if not response.ok:
        raise Exception(response.text)


def post_question_prediction(
    question_id: int, prediction_percentage: float, metaculus_token: str
) -> None:
    """
    Post a prediction value (between 1 and 100) on the question.
    """
    assert 1 <= prediction_percentage <= 100, "Prediction must be between 1 and 100"
    auth_headers = {"headers": {"Authorization": f"Token {metaculus_token}"}}
    url = f"{API_BASE_URL}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        **auth_headers,
    )
    if not response.ok:
        raise Exception(response.text)


def get_question_details(question_id: int, metaculus_token: str) -> dict:
    """
    Get all details about a specific question.
    """
    auth_headers = {"headers": {"Authorization": f"Token {metaculus_token}"}}
    url = f"{API_BASE_URL}/questions/{question_id}/"
    response = requests.get(
        url,
        **auth_headers,
    )
    if not response.ok:
        raise Exception(response.text)
    return json.loads(response.content)


def list_questions(
    tournament_id: int,
    metaculus_token: str,
    offset=0,
    count=10,
) -> list[dict]:
    """
    List (all details) {count} questions from the {tournament_id}
    """
    auth_headers = {"headers": {"Authorization": f"Token {metaculus_token}"}}
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": "binary",
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/questions/"
    response = requests.get(url, **auth_headers, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data
