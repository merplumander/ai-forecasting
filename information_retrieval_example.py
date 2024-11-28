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
from src.query.language_models import OpenAIModel
from src.query.PromptBuilder import BinaryQuestionWithDescriptionAndNewsPromptBuilder

load_dotenv(".env")
# %%
dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_08_01_to_election_",
    download_new_data=False,
)
question_ids = dataset.questions.index.tolist()
question = dataset.questions.loc[question_ids[19]]
print(question.title)
# %%
queries = generate_search_queries(question)
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
model = OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o")
resp = model.query_model(
    BinaryQuestionWithDescriptionAndNewsPromptBuilder.get_user_prompt(question),
    BinaryQuestionWithDescriptionAndNewsPromptBuilder.get_system_prompt(),
)

# %%
