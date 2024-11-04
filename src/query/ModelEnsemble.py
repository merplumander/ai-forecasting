import concurrent
from typing import List, Tuple

import numpy as np


class ModelEnsemble:
    def __init__(self, models):
        self.models = models

    def make_forecast(
        self,
        forecasting_question: str,
        context: str,
        use_probabilities: bool = False,
    ) -> Tuple[List[int], List[str]]:
        """Make a forecast using the ensemble of models.

        Parameters
        ----------
        forecasting_question : str
            Question to be forecasted.
        context : str
            System prompt and context.
        use_probabilities : bool, optional
            currently not implemented, by default False

        Returns
        -------
        Tuple[List[int], List[str]]
            Tuple with forecasted answers and explanations.
        """
        if use_probabilities:
            raise NotImplementedError(
                "Using probabilities is not implemented for model ensembles."
            )
        predicted_probabilities = []
        explanations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    model.make_forecast,
                    forecasting_question,
                    context,
                )
                for model in self.models
            ]
        for future in futures:
            result = future.result()
            predicted_probabilities.append(result[0])
            explanations.append(result[1])
        return (predicted_probabilities, explanations)
