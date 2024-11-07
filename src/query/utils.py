import re
from functools import wraps

import numpy as np


def extract_probability(reply):
    probability = None
    match = re.search(r"Answer: (\d+)", reply)
    if match:
        probability = int(match.group(1))
    else:
        raise ValueError("The model did not return a valid answer.")
    return probability


def retry_on_model_failure(max_retries=3):
    """Retry querying a model a specified number of times before giving up."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except ValueError as e:
                    attempts += 1
                    print(
                        f"Attempt {attempts} of model"
                        f" {args[0].model_version} failed with error: {e}"
                    )
            print(f"Model {args[0].model_version} failed after {max_retries} attempts.")
            return None

        return wrapper

    return decorator


def aggregate_forecasting_explanations(
    question, reasonings, probabilities, language_model
):
    lower, median, upper = np.quantile(probabilities, [0.05, 0.50, 0.95])
    confidence_range_string = f"[{lower}, {upper}]"
    newline = "\n"
    reasonings = f"{newline.join(
        f"Reasoning {number}:\n{reasoning}\n"
        for number, reasoning in enumerate(reasonings))}"
    with open("combine_reasoning_prompt.txt", "r") as file:
        # Read the entire content of the file
        combine_prompt = file.read()

    combine_prompt = combine_prompt.format(
        question=question,
        reasonings=reasonings,
        median_forecast=median,
        confidence_range_string=confidence_range_string,
    )
    aggregated_explanation = language_model.query_model(
        user_prompt=combine_prompt, system_prompt="None"
    )
    return aggregated_explanation
