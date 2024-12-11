# %%
import os
import re
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from src.query.language_models import (
    AnthropicModel,
    GeminiModel,
    LLAMAModel,
    OpenAIModel,
    XAIModel,
)
from src.query.ModelEnsemble import ModelEnsemble
from src.utils import get_project_root

ROOT = get_project_root()


def validate_asterisk_number(text):
    # Regular expression to match **number** where the number is between 0 and
    # 100
    match = re.fullmatch(r"\*\*(\d{1,3})\*\*", text)

    # Check if the regex pattern matched and the number is in the range 0-100
    if match:
        number = int(match.group(1))
        if 0 <= number <= 100:
            return number
        else:
            raise ValueError(f"The number {number} is out of the valid range (0-100).")
    else:
        raise ValueError(
            "The input string does not match the required format (**number**)."
        )


# %%
load_dotenv(".env")

# %%
forecasting_question = (
    "Will global average temperatures be more than 2 degrees Celsius above"
    " pre-industrial levels by the year 2050?"
)

# %%
with open(
    ROOT / "prompts" / "binary_question_with_description_system_prompt.txt", "r"
) as file:
    # Read the entire content of the file
    context_prompt = file.read()
# %%
open_ai_model = OpenAIModel(api_key=os.environ.get("OPENAI_API_KEY"))
response = open_ai_model.make_forecast(
    forecasting_question,
    context_prompt,
)
print(response)

# %%
anthropic_model = AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"))
response = anthropic_model.make_forecast(
    forecasting_question,
    context_prompt,
)
print(response)
# %%
gemini_model = GeminiModel(os.environ.get("GEMINI_API_KEY"))
response = gemini_model.make_forecast(
    forecasting_question,
    context_prompt,
)
print(response)
# %%
xai_model = XAIModel(os.environ.get("XAI_API_KEY"))
response = xai_model.make_forecast(forecasting_question, context_prompt)
print(response)
# %%
llama_model = LLAMAModel(os.environ.get("LLAMA_API_KEY"))
response = llama_model.make_forecast(forecasting_question, context_prompt)
print(response)

# %%
open_ai_model = OpenAIModel(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_model = AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"))
gemini_model = GeminiModel(os.environ.get("GEMINI_API_KEY"))
xai_model = XAIModel(os.environ.get("XAI_API_KEY"))
llama_model = LLAMAModel(os.environ.get("LLAMA_API_KEY"))
ensemble = ModelEnsemble([anthropic_model, gemini_model, llama_model])
# %%
ensemble_responses = ensemble.make_forecast(forecasting_question, context_prompt)

# %%
probabilities = [
    probability for probability in ensemble_responses[0] if probability is not None
]
lower, median, upper = np.quantile(probabilities, [0.05, 0.50, 0.95])
for i in range(len(ensemble_responses[1])):
    explanation = ensemble_responses[1][i]
    print(explanation)

# %%
reasonings = ensemble_responses[1]
median_forecast = median
confidence_range_string = f"[{lower}, {upper}]"
newline = "\n"
reasonings = f"{newline.join(
    f"Reasoning {number}:\n{reasoning}\n"
    for number, reasoning in enumerate(reasonings))}"
with open("combine_reasoning_prompt.txt", "r") as file:
    # Read the entire content of the file
    combine_prompt = file.read()

combine_prompt = combine_prompt.format(
    question=forecasting_question,
    reasonings=reasonings,
    median_forecast=median_forecast,
    confidence_range_string=confidence_range_string,
)

print(combine_prompt)
# %%
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"{combine_prompt}"},
        # {"role": "system", "content": f"{context_prompt}"},
    ],
    max_tokens=1000,
    temperature=0,
)
# %%
from src.query.utils import aggregate_forecasting_explanations

# %%
language_model = OpenAIModel(
    api_key=os.environ.get("OPENAI_API_KEY"), model_version="gpt-4o"
)
response = aggregate_forecasting_explanations(
    question=forecasting_question,
    reasonings=reasonings,
    probabilities=probabilities,
    language_model=language_model,
)
# %%

# %%
