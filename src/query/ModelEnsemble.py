import concurrent
from typing import List, Tuple, Type

from src.dataset.dataset import Question
from src.query.language_models import LanguageModel
from src.query.PromptBuilder import PromptBuilder


class ModelEnsemble:
    def __init__(self, models: List[LanguageModel]):
        self.models = models

    def make_forecast_from_question(
        self,
        question: Question,
        prompt_builder: Type[PromptBuilder],
        model_query_repeats=3,
    ) -> List[Tuple[str, str, int, str]]:
        """Make a forecast using the ensemble of models.

        Parameters
        ----------
        question : Question
            Question to be forecasted.
        prompt_builder : Type[PromptBuilder]
            Prompt builder to be used.
        model_query_repeats : int, optional
            How many times the prompt should be repeated for each model, by default 3

        Returns
        -------
        List[Tuple[str, str, int, str]]
            List of tuples containing the question id, model name, forecasted answer and explanation.
        """
        query_list = [
            model for model in self.models for _ in range(model_query_repeats)
        ]
        system_prompt = prompt_builder.get_system_prompt()
        user_prompt = prompt_builder.get_user_prompt(question)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    model.make_forecast,
                    user_prompt,
                    system_prompt,
                )
                for model in query_list
            ]
        results = []
        for i, future in enumerate(futures):
            result = future.result()
            if result is not None:
                results.append(
                    (
                        question.question_id,
                        query_list[i].model_version,
                        result[0],
                        result[1],
                    )
                )
        return results

    def make_forecast(
        self,
        forecasting_question: str,
        context: str,
        model_query_repeats=3,
        use_probabilities: bool = False,
    ) -> List[Tuple[str, int, str]]:
        """Make a forecast using the ensemble of models.

        Parameters
        ----------
        forecasting_question : str
            Question to be forecasted.
        context : str
            System prompt and context.
        model_query_repeats : int, optional
            How many times the prompt should be repeated for each model, by default 3
        use_probabilities : bool, optional
            currently not implemented, by default False

        Returns
        -------
        List[Tuple[str, int, str]]
            List of tuples containing the model name, forecasted answer and explanation.
        """
        if use_probabilities:
            raise NotImplementedError(
                "Using probabilities is not implemented for model ensembles."
            )
        query_list = [
            model for model in self.models for _ in range(model_query_repeats)
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    model.make_forecast,
                    forecasting_question,
                    context,
                )
                for model in query_list
            ]
        results = []
        for i, future in enumerate(futures):
            result = future.result()
            results.append((query_list[i].model_version, result[0], result[1]))
        return results
