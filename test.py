# %%
import os
import time

from dotenv import load_dotenv

from src.query.language_models import (
    AnthropicModel,
    GeminiModel,
    LLAMAModel,
    MistralModel,
    OpenAIModel,
    QwenModel,
    XAIModel,
)
from src.query.ModelEnsemble import ModelEnsemble

# %%
load_dotenv(".env")
forecasting_question = (
    "Will global average temperatures be more than 2 degrees Celsius above"
    " pre-industrial levels by the year 2050?"
)
with open("context_prompt_logprobs.txt", "r") as file:
    # Read the entire content of the file
    context_prompt = file.read()

# %%
# %%
open_ai_model = OpenAIModel(os.environ.get("OPENAI_API_KEY"), "o1-mini")
# anthropic_model = AnthropicModel(
#     os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022"
# )
# gemini_model = GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro")
# # xai_model = XAIModel(os.environ.get("XAI_API_KEY"))
# llama_model = LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.1-405b")
# ensemble = ModelEnsemble([open_ai_model, anthropic_model, gemini_model, llama_model])
open_ai_model = OpenAIModel(os.environ.get("OPENAI_API_KEY"))
anthropic_model = AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"))
gemini_model = GeminiModel(os.environ.get("GEMINI_API_KEY"))
xai_model = XAIModel(os.environ.get("XAI_API_KEY"))
llama_model = LLAMAModel(os.environ.get("LLAMA_API_KEY"))
mistral_model = MistralModel(os.environ.get("MISTRAL_API_KEY"))
qwen_model = QwenModel(os.environ.get("DASHSCOPE_API_KEY"))
ensemble = ModelEnsemble([open_ai_model, anthropic_model, gemini_model, llama_model])
# %%
start_time = time.time()
ensemble_responses = ensemble.make_forecast(forecasting_question, context_prompt)
end_time = time.time()

print(f"Time taken for ensemble forecast: {end_time - start_time} seconds")
print(ensemble_responses)
