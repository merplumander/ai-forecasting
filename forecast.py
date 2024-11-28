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

ROOT = Path(__file__).parent


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
ensemble = ModelEnsemble(
    [open_ai_model, anthropic_model, gemini_model, xai_model, llama_model]
)
# %%
ensemble_responses = ensemble.make_forecast(forecasting_question, context_prompt)
