import os
import re
from typing import List

from newspaper.article import Article
from tqdm import tqdm

from src.dataset.dataset import Question
from src.query.language_models import GeminiModel, LanguageModel
from src.query.PromptBuilder import (
    ArticleRelevancyPromptBuilder,
    NewsRetrievalPromptBuilder,
)
from src.query.utils import retry_on_model_failure


def generate_search_queries(
    question: Question,
    language_model: LanguageModel = None,
    num_queries: int = 10,
    max_query_words: int = 10,
    include_question: bool = True,
) -> List[str]:
    """Generates search queries for a given question using a language model.

    Parameters
    ----------
    question : Question
    language_model : LanguageModel, optional
        If no language model is provided, one is picked by the function, by default None
    num_queries : int, optional
        Number of queries to generate, by default 10
    max_query_words : int, optional
        Maximum number of words per query, by default 10
    include_question : bool, optional
        Whether to include the original question in the queries, by default True

    Returns
    -------
    List[str]
        Queries generated by the language model
    """
    assert num_queries > 0, "Number of queries must be greater than 0"
    assert max_query_words > 0, "Number of words per query must be greater than 0"

    if language_model is None:
        language_model = GeminiModel(
            os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"
        )
    system_prompt = NewsRetrievalPromptBuilder.get_system_prompt(max_query_words)
    user_prompt = NewsRetrievalPromptBuilder.get_user_prompt(question, num_queries)

    @retry_on_model_failure(max_retries=3)
    def get_queries(language_model, user_prompt, system_prompt):
        # This regular expression pattern is used to match a query in the
        # following format:
        # - A sequence of one or more digits (\d+)
        # - Followed by a period and a space (\.\s+)
        # - Followed by any characters (non-greedy match) until one of the
        #   following is encountered: a semicolon, a period, a newline, or the
        #   end of the string
        query_pattern = r"\d+\.\s+(.*?)(?:[;.]|\n|$)"
        response = language_model.query_model(user_prompt, system_prompt)
        queries = re.findall(query_pattern, response)
        if len(queries) != num_queries:
            raise ValueError(
                f"The model did only return {len(queries)} and not"
                f" {num_queries} queries."
            )
        return queries

    queries = get_queries(language_model, user_prompt, system_prompt)
    if include_question:
        queries.append(question.title)
    return queries


def rate_article_relevancy(
    article: Article,
    question: Question,
    article_cutoff: int = 1000,
    language_model: LanguageModel = None,
) -> float:
    """Rates the relevancy of an article to a question using a language model.

    Parameters
    ----------
    article : Article
    question : Question
    article_cutoff: int, optional
        Maximum number of chars to use from the article, by default 1000
    language_model : LanguageModel, optional
        If no language model is provided, one is picked by the function, by
        default None

    Returns
    -------
    float
        Relevancy score
    """
    # TODO add today date and article published date to the prompt
    system_prompt = ArticleRelevancyPromptBuilder.get_system_prompt()
    user_prompt = ArticleRelevancyPromptBuilder.get_user_prompt(
        question, article, article_cutoff
    )
    if language_model is None:
        language_model = GeminiModel(
            os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"
        )

    @retry_on_model_failure(max_retries=3)
    def rate_relevancy(user_prompt, system_prompt):
        response = language_model.query_model(user_prompt, system_prompt)
        match = re.search(r"Rating: (\d+)", response)
        if match and int(match.group(1)) in range(1, 7):
            return int(match.group(1))
        else:
            raise ValueError("The model did not return a valid answer.")

    return rate_relevancy(user_prompt, system_prompt)


def get_relevant_articles(
    articles: List[Article],
    question: Question,
    n: int = 5,
    min_score: float = 4.0,
    article_cutoff: int = 1000,
    language_model: LanguageModel = None,
) -> List[Article]:
    """Gets the most relevant articles based on a relevancy score.

    Parameters
    ----------
    articles : List[Article]
        List of articles to evaluate
    question : Question
        The question to evaluate the articles against
    n : int, optional
        Number of top relevant articles to return, by default 5
        If there are less than n articles with a score >= min_score, all
        articles are returned.
    min_score : float, optional
        Minimum relevancy score to consider an article relevant, by default 4.0
    article_cutoff: int, optional
        Maximum number of chars to use from the articles, by default 1000
    language_model : LanguageModel, optional
        If no language model is provided, one is picked by the function, by
        default None

    Returns
    -------
    List[Article]
        List of relevant articles
    """
    scored_articles = []
    for article in tqdm(articles):
        score = rate_article_relevancy(
            article, question, article_cutoff, language_model
        )
        if score >= min_score:
            scored_articles.append((article, score))

    if len(scored_articles) == 0:
        raise ValueError("No relevant articles found. Try lowering the min_score.")
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    return [article for article, _ in scored_articles[:n]]
