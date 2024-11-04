import re
from functools import wraps


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
