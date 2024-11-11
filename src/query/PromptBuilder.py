from abc import ABC, abstractmethod
from pathlib import Path

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
