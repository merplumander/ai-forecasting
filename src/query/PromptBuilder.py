from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, override

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

    def get_default_system_prompt_id() -> str:
        """Returns the default system prompt id."""
        return "sp_0"


class BinaryQuestionWithDescriptionPromptBuilder(PromptBuilder):

    def get_system_prompt(prompt_id: str = None):
        if prompt_id is None:
            prompt_id = (
                BinaryQuestionWithDescriptionPromptBuilder.get_default_system_prompt_id()
            )
        with open(ROOT / "prompts" / "bqdsp" / f"{prompt_id}.txt", "r") as file:
            system_prompt = file.read()
        with open(
            ROOT / "prompts" / "bqdsp" / "response_format_prompt.txt", "r"
        ) as file:
            format_prompt = file.read()
        return "\n\n".join([system_prompt, format_prompt])

    def get_user_prompt(question: BinaryQuestion):
        with open(
            ROOT / "prompts" / "binary_question_with_description_user_prompt.txt", "r"
        ) as file:
            user_prompt = file.read()
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            today_date=datetime.today().strftime("%b %d, %Y"),
        )
        return user_prompt

    @override
    def get_default_system_prompt_id() -> str:
        """Returns the default system prompt id."""
        return "bqdsp_0"


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
            today_date=datetime.today().strftime("%b %d, %Y"),
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
            today_date=datetime.today().strftime("%b %d, %Y"),
            article_publication_date=article.publish_date.strftime("%b %d, %Y"),
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
        articles_text = "\n\n".join(
            [
                f"Articel {i+1} (published on"
                f" {article.publish_date.strftime('%b %d, %Y')}):\n{article.text.replace('\n\n', '\n')}"
                for i, article in enumerate(articles)
            ]
        )
        user_prompt = user_prompt.format(
            question_title=question.title,
            question_description=question.description,
            articles=articles_text,
        )
        return user_prompt
