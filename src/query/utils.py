import re


def extract_probability(reply):
    probability = None
    match = re.search(r"Answer: (\d+)", reply)
    if match:
        probability = int(match)
    else:
        raise ValueError("The model did not return a valid answer.")
    return probability
