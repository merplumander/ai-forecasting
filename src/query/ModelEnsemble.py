import concurrent
from typing import List, Tuple, Type

from src.dataset.dataset import EnsembleForecast, Question
from src.query.language_models import LanguageModel
from src.query.PromptBuilder import PromptBuilder


class ModelEnsemble:
    def __init__(self, models: List[LanguageModel]):
        self.models = models

    def make_forecast_from_question(
        self,
        question: Question,
        prompt_builder: Type[PromptBuilder],
        model_query_repeats: int = 1,
        system_prompt_ids: List[str] = None,
    ) -> EnsembleForecast:
        """Make a forecast using the ensemble of models.

        This method generates forecasts for a given question using each model in
        the ensemble. It can use multiple system prompts and repeat queries to
        generate a set of forecasts for each model.

        Parameters
        ----------
        question : Question
            Question to be forecasted.
        prompt_builder : Type[PromptBuilder]
            Prompt builder to be used for generating model prompts.
        model_query_repeats : int, optional
            How many times the prompt should be repeated for each model, by
            default 1
        system_prompt_ids : List[str], optional
            List of different system prompt ids to be used, by default None

        Returns
        -------
        EnsembleForecast
            A collection of forecasts from all models in the ensemble.
        """
        assert model_query_repeats > 0
        assert (
            model_query_repeats == 1 or system_prompt_ids is None
        ), "User prompt ids can only be used when model_query_repeats is 1."
        if system_prompt_ids is None:
            query_list = [
                model for model in self.models for _ in range(model_query_repeats)
            ]
            system_prompt = prompt_builder.get_system_prompt()
            default_system_prompt_id = prompt_builder.get_default_system_prompt_id()
            system_prompt_ids_rep = [default_system_prompt_id for _ in query_list]
            system_prompts = [system_prompt for _ in query_list]
        else:
            query_list = [model for model in self.models for _ in system_prompt_ids]
            system_prompt_ids_rep = [
                prompt for _ in self.models for prompt in system_prompt_ids
            ]
            system_prompts = [
                prompt_builder.get_system_prompt(prompt_id)
                for prompt_id in system_prompt_ids_rep
            ]
        user_prompt = prompt_builder.get_user_prompt(question)
        user_prompts = [user_prompt for _ in query_list]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda args: args[0].make_forecast(*args[1:]),
                    zip(query_list, user_prompts, system_prompts),
                )
            )
            if system_prompt_ids is not None:
                for i, forecast in enumerate(results):
                    forecast.prompt_id = system_prompt_ids_rep[i]
        forecast = EnsembleForecast(
            forecasts=results,
            question_id=question.question_id,
        )
        return forecast

    def make_forecast(
        self,
        forecasting_question: str,
        context: str,
        model_query_repeats=3,
        use_probabilities: bool = False,
    ) -> EnsembleForecast:
        """Make a forecast using the ensemble of models.

        This method is a simpler version of make_forecast_from_question that takes a raw
        question string and context instead of a Question object.

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
        EnsembleForecast
            A collection of forecasts from all models in the ensemble.
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
            forecast = future.result()
            forecast.model = query_list[i].model_version
            results.append((forecast))
        forecast = EnsembleForecast(
            forecasts=results,
        )
        return forecast
