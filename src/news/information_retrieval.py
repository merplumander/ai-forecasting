import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

from newspaper.article import Article
from tqdm import tqdm

from src.dataset.dataset import Question
from src.news.news_api import get_gnews_articles, retrieve_gnews_articles_fulldata
from src.query.language_models import GeminiModel, LanguageModel
from src.query.PromptBuilder import (
    ArticleRelevancyPromptBuilder,
    ArticlesSummaryPromptBuilder,
    NewsRetrievalPromptBuilder,
)
from src.query.utils import retry_on_model_failure
from src.utils import logger


def search_web_and_summarize(
    question: Question,
    language_model: LanguageModel = None,
    num_search_queries: int = 5,
    max_words_per_query: int = 10,
    include_question_as_query: bool = True,
    max_results_per_query=20,
    max_n_relevant_articles=10,
):
    queries = generate_search_queries(
        question,
        language_model=language_model,
        num_queries=num_search_queries,
        max_query_words=max_words_per_query,
        include_question=include_question_as_query,
    )
    relevant_articles = []
    results_per_query = 5
    while (
        len(relevant_articles) < max_n_relevant_articles
        and results_per_query <= max_results_per_query
    ):
        articles = get_gnews_articles(queries, max_results=results_per_query)
        relevant_urls = [a.url for a in relevant_articles]
        articles = [
            list(
                filter(
                    lambda article: article["url"] not in relevant_urls, article_list
                )
            )
            for article_list in articles
        ]
        full_articles = retrieve_gnews_articles_fulldata(
            articles, num_articles=results_per_query
        )
        relevant_articles += get_relevant_articles(
            full_articles,
            question,
            n=(max_n_relevant_articles - len(relevant_articles)),
            language_model=language_model,
        )
        results_per_query *= 2
    summary = summarize_articles_for_question(
        relevant_articles, question, language_model=language_model
    )

    return summary


def search_web_and_summarize_parallel(
    questions: List[Question],
    language_model: LanguageModel = None,
    num_search_queries: int = 5,
    max_words_per_query: int = 10,
    include_question_as_query: bool = True,
    max_results_per_query=10,
    max_n_relevant_articles=10,
) -> List[str]:
    """Same as search_web_and_summarize but parallelized for multiple questions"""
    args_list = [
        (
            question,
            language_model,
            num_search_queries,
            max_words_per_query,
            include_question_as_query,
            max_results_per_query,
            max_n_relevant_articles,
        )
        for question in questions
    ]
    with ThreadPoolExecutor(max_workers=50) as executor:
        summaries = list(
            executor.map(lambda args: search_web_and_summarize(*args), args_list)
        )

    return summaries


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
        # Strip stars which are sometimes added to the queries
        queries = [query.strip("*") for query in queries]
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
    article_save_path: str = None,
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
    # check if save file exists. If yes load from file
    if article_save_path is not None and os.path.exists(article_save_path):
        with open(article_save_path, "r") as file:
            scored_articles = json.load(file)
            scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [article for article, _ in scored_articles[:n]]

    scored_articles = []
    for article in tqdm(articles):
        score = rate_article_relevancy(
            article, question, article_cutoff, language_model
        )
        if score >= min_score:
            scored_articles.append((article, score))

    if len(scored_articles) == 0:
        raise ValueError("No relevant articles found. Try lowering the min_score.")
    if article_save_path is not None:
        with open(article_save_path, "w") as file:
            json.dump(scored_articles, file)
    scored_articles.sort(key=lambda x: x[1], reverse=True)

    return [article for article, _ in scored_articles[:n]]


def summarize_articles_for_question(
    articles: List[Article], question: Question, language_model: LanguageModel = None
) -> str:
    """Summarizes a list of articles based on a question.

    Parameters
    ----------
    articles : List[Article]
        List of articles to summarize
    question : Question
    language_model : LanguageModel, optional
        If no language model is provided, one is picked by the function, by
        default None

    Returns
    -------
    str
        Summary of the articles
    """
    if language_model is None:
        language_model = GeminiModel(
            os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"
        )

    system_prompt = ArticlesSummaryPromptBuilder.get_system_prompt()
    user_prompt = ArticlesSummaryPromptBuilder.get_user_prompt(question, articles)

    logger.info(f"System Prompt: {system_prompt}")
    logger.info(f"User Prompt: {user_prompt}")

    @retry_on_model_failure(max_retries=3)
    def get_summary(language_model, user_prompt, system_prompt):
        response = language_model.query_model(
            user_prompt, system_prompt, max_output_tokens=10000
        )

        logger.info(f"\n\n------------------LLM RESPONSE------------\n\n{response}")
        summary = re.search(r"Summary:\s*(.*)", response, re.DOTALL)
        if summary:
            return summary.group(1).strip()
        else:
            raise ValueError("The model did not return a valid summary.")

    return get_summary(language_model, user_prompt, system_prompt)
