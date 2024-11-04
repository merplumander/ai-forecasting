import concurrent

import numpy as np


class ModelEnsemble:
    def __init__(self, models):
        self.models = models

    def make_forecast(
        self, forecasting_question: str, context: str, use_probabilities=False
    ):
        if use_probabilities:
            raise NotImplementedError(
                "Using probabilities is not implemented for model ensembles."
            )
        predicted_probabilities = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    model.make_forecast_retry, forecasting_question, context
                )
                for model in self.models
            ]
        for future in futures:
            predicted_probabilities.append(future.result())
        return predicted_probabilities
