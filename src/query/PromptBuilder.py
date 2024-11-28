from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from newspaper import Article

from src.dataset.dataset import BinaryQuestion, Question

ROOT = Path(__file__).parent.parent.parent


class PromptBuilder(ABC):

    @abstractmethod
    def get_system_prompt() -> str:
        """Returns the system prompt."""
        pass

    @abstractmethod
    def get_user_prompt(question: Question) -> str:
        """Returns the user prompt for the specified question."""
        pass


class BinaryQuestionWithDescriptionPromptBuilder(PromptBuilder):

    def get_system_prompt():
        with open(
            ROOT / "prompts" / "binary_question_with_description_system_prompt.txt", "r"
        ) as file:
            system_prompt = file.read()
        return system_prompt

    def get_user_prompt(question: BinaryQuestion):
        with open(
            ROOT / "prompts" / "binary_question_with_description_user_prompt.txt", "r"
        ) as file:
            user_prompt = file.read()
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
        )
        return user_prompt


class BinaryQuestionWithDescriptionAndNewsPromptBuilder(
    BinaryQuestionWithDescriptionPromptBuilder
):

    def get_user_prompt(question: BinaryQuestion):
        with open(
            ROOT
            / "prompts"
            / "binary_question_with_description_and_news_user_prompt.txt",
            "r",
        ) as file:
            user_prompt = file.read()
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            news_summary=question.news_summary,
        )
        return user_prompt


class NewsRetrievalPromptBuilder(PromptBuilder):

    def get_system_prompt(max_query_words=10):
        with open(ROOT / "prompts" / "news_retrieval_system_prompt.txt", "r") as file:
            system_prompt = file.read()
        system_prompt = system_prompt.format(max_words=max_query_words)
        return system_prompt

    def get_user_prompt(question: Question, num_queries=10):
        with open(ROOT / "prompts" / "news_retrieval_user_prompt.txt", "r") as file:
            user_prompt = file.read()
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            num_queries=num_queries,
        )
        return user_prompt


class ArticleRelevancyPromptBuilder(PromptBuilder):

    def get_system_prompt():
        with open(
            ROOT / "prompts" / "article_relevancy_system_prompt.txt", "r"
        ) as file:
            system_prompt = file.read()
        return system_prompt

    def get_user_prompt(question: Question, article: Article, article_cutoff=1000):
        with open(ROOT / "prompts" / "article_relevancy_user_prompt.txt", "r") as file:
            user_prompt = file.read()
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            article=article.text[:article_cutoff],
            article_cutoff=article_cutoff,
        )
        return user_prompt


class ArticlesSummaryPromptBuilder(PromptBuilder):

    def get_system_prompt():
        with open(ROOT / "prompts" / "articles_summary_system_prompt.txt", "r") as file:
            system_prompt = file.read()
        return system_prompt

    def get_user_prompt(question: Question, articles: List[Article]):
        with open(ROOT / "prompts" / "articles_summary_user_prompt.txt", "r") as file:
            user_prompt = file.read()
        articles_text = "\n\n".join([article.text for article in articles])
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            articles=articles_text,
        )
        return user_prompt
