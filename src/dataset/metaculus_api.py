# %%
from typing import List

import pandas as pd
import requests


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
        print("Error:", response.status_code)
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
        print("Downloading posts:", offset, "to", limit)
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
            print(question)
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
    print("Total questions downloaded:", len(questions_df))
    return pd.DataFrame(questions_df)
