# %%
import os
from datetime import datetime

from dotenv import load_dotenv

from src.dataset.dataset import BinaryQuestion, MetaculusDataset
from src.dataset.metaculus_api import get_question_details
from src.news.information_retrieval import (
    generate_search_queries,
    get_relevant_articles,
    summarize_articles_for_question,
)
from src.news.news_api import get_gnews_articles, retrieve_gnews_articles_fulldata
from src.query.language_models import (
    AnthropicModel,
    GeminiModel,
    OpenAIModel,
    QwenModel,
)
from src.query.ModelEnsemble import ModelEnsemble
from src.query.PromptBuilder import BinaryQuestionWithDescriptionAndNewsPromptBuilder
from src.utils import get_project_root

load_dotenv(".env")
# %%
# assumes questions are from Q4 of tournament
ROOT = get_project_root()
article_folder = ROOT / "forecasts" / "metaculus-tournament" / "Q4"


dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_08_01_to_election_",
    download_new_data=False,
)
question_ids = dataset.questions.index.tolist()
question = dataset.get_question(question_ids[19])
print(question.title)

# %%
METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN")
question_id = 30880
question_details = get_question_details(question_id, metaculus_token=METACULUS_TOKEN)
date_string = question_details["created_at"]
created_at = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
resolved = question_details["status"] == "resolved"

title = question_details["question"]["title"]
resolution_criteria = question_details["question"]["resolution_criteria"]
background = question_details["question"]["description"]
fine_print = question_details["question"]["fine_print"]
description = (
    f"Background: \n{background}\n\n"
    f"Fine Print: \n{fine_print}\n\n"
    f"Resolution Criteria: \n{resolution_criteria}\n\n"
)
question = BinaryQuestion(
    question_id=question_id,
    title=title,
    created_at=created_at,
    resolved=resolved,
    description=description,
)
# %%
queries = generate_search_queries(question, num_queries=5)
print(queries)
# %%
articles = get_gnews_articles(queries)
# %%
full_articles = retrieve_gnews_articles_fulldata(articles, num_articles=10)

# %%
article_save_path = article_folder / f"metaculus-{question.question_id}-news.txt"
relevant_articles = get_relevant_articles(
    full_articles, question, n=10, article_save_path=article_save_path, min_score=4
)
# %%
for article in relevant_articles:
    print(article.title)
    print(article.text)
    print("\n\n\n\n------------------------------------------\n\n\n\n")
# %%
summary = summarize_articles_for_question(relevant_articles, question)
# %%
question.news_summary = summary
# %%
ensemble_models = [
    QwenModel(os.environ.get("DASHSCOPE_API_KEY"), "qwen-max"),
]
ensemble = ModelEnsemble(ensemble_models)

# %%
# bqdsp: Binary Question with Description System Prompt
# we have a number of different system prompts to increase diversity amonng the
# predictions and to be able to do an evolutionary prompt improvement
system_prompt_ids = ["bqdsp_0"]  # , "bqdsp_1", "bqdsp_2", "bqdsp_3"]
ensemble_responses = ensemble.make_forecast_from_question(
    question,
    BinaryQuestionWithDescriptionAndNewsPromptBuilder,
    system_prompt_ids=system_prompt_ids,
)


# %%
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame(
    ensemble_responses,
    columns=["question_id", "model_name", "prompt_id", "forecast", "explanation"],
)
plt.scatter(data.model_name, data.forecast, alpha=0.2)
# %%
