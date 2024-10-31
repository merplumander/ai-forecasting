import numpy as np


class ModelEnsample:
    def __init__(self, models):
        self.models = models

    def make_forecast(
        self, forecasting_question: str, context: str, use_probabilities=False
    ):
        if use_probabilities:
            raise NotImplementedError(
                "Using probabilities is not implemented for model ensamples."
            )
        predicted_probabilities = []
        for model in self.models:
            model_prediction = model.make_forecast(
                forecasting_question, context, use_probabilities
            )
            predicted_probabilities.append(model_prediction)
        return predicted_probabilities
