# %%
import os

from dotenv import load_dotenv

from src.dataset.dataset import MetaculusDataset
from src.news.information_retrieval import (
    generate_search_queries,
    get_relevant_articles,
    summarize_articles_for_question,
)
from src.news.news_api import get_gnews_articles, retrieve_gnews_articles_fulldata
from src.query.language_models import AnthropicModel, GeminiModel, OpenAIModel
from src.query.ModelEnsemble import ModelEnsemble
from src.query.PromptBuilder import BinaryQuestionWithDescriptionAndNewsPromptBuilder

load_dotenv(".env")
# %%
dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_08_01_to_election_",
    download_new_data=False,
)
question_ids = dataset.questions.index.tolist()
question = dataset.get_question(question_ids[19])
print(question.title)
# %%
queries = generate_search_queries(question, num_queries=10)
print(queries)
# %%
articles = get_gnews_articles(queries)
# %%
full_articles = retrieve_gnews_articles_fulldata(articles, num_articles=2)
# %%
relevant_articles = get_relevant_articles(full_articles, question, n=8)
# %%
summary = summarize_articles_for_question(relevant_articles, question)
# %%
question.news_summary = summary
# %%
ensemble_models = [
    OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o"),
    AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022"),
    GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"),
]
ensemble = ModelEnsemble(ensemble_models)

# %%
system_prompt_ids = ["bqdsp_0", "bqdsp_1", "bqdsp_2", "bqdsp_3"]
ensemble_responses = ensemble.make_forecast_from_question(
    question,
    BinaryQuestionWithDescriptionAndNewsPromptBuilder,
    system_prompt_ids=system_prompt_ids,
)

import matplotlib.pyplot as plt

# %%
import pandas as pd

data = pd.DataFrame(
    ensemble_responses,
    columns=["question_id", "model_name", "prompt_id", "forecast", "explanation"],
)
plt.scatter(data.model_name, data.forecast, alpha=0.2)
# %%
