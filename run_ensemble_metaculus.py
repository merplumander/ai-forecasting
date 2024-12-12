# %%
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.analysis.utils import load_raw_forecasting_data
from src.dataset.dataset import MetaculusDataset
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
from src.query.PromptBuilder import BinaryQuestionWithDescriptionPromptBuilder
from src.utils import ROOT

# %%
dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_08_01_to_election_",
    download_new_data=False,
)
# dataset.select_questions_with_status("resolved")
# dataset.select_questions_with_forecast_type("binary")
# dataset.select_questions_newer_than(datetime(year=2024, month=8, day=1))
# dataset.select_questions_resolved_before(datetime(year=2024, month=11, day=5))

# dataset.file_infix = "resolved_binary_from_2024_08_01_to_election_"
# dataset.save()

# %%
load_dotenv(".env")
ensemble_models = [
    OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o"),
    OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o-mini"),
    OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4-turbo"),
    AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022"),
    AnthropicModel(os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-haiku-20241022"),
    GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"),
    GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-flash-001"),
    LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.1-405b"),
    LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.2-90b-vision"),
    LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.2-11b-vision"),
    MistralModel(os.environ.get("MISTRAL_API_KEY"), "mistral-large-2407"),
    MistralModel(os.environ.get("MISTRAL_API_KEY"), "mistral-small-2402"),
    XAIModel(os.environ.get("XAI_API_KEY"), "grok-beta"),
    #     QwenModel(os.environ.get("DASHSCOPE_API_KEY"), "qwen-max"),
    #     QwenModel(os.environ.get("DASHSCOPE_API_KEY"), "qwen-plus"),
]
ensemble = ModelEnsemble(ensemble_models)
# %%
question_ids = dataset.questions.index.tolist()
question_ids.sort()
question_ids = question_ids[::2]
# %%
for question_id in tqdm(question_ids[0:60]):
    question = dataset.get_question(question_id)
    forecasts = ensemble.make_forecast_from_question(
        question, BinaryQuestionWithDescriptionPromptBuilder, model_query_repeats=20
    )

    with open(f"forecasts/{question_id}.txt", "a") as file:
        file.writelines(f"{str(forecast)}\n" for forecast in forecasts)
# %%
# for each model look through the data and if there are less than 20 forecasts for a question, make more
data = load_raw_forecasting_data(ROOT / "forecasts")
models = [GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001")]
ensemble = ModelEnsemble(models)
counts = data.groupby(["model_name", "question_id"]).agg("count")
for question_id in counts.loc["gemini-1.5-pro-001"].index:
    count = counts.loc["gemini-1.5-pro-001", question_id].forecast
    if count < 20:
        print(count)
        forecasts = ensemble.make_forecast_from_question(
            dataset.get_question(question_id),
            BinaryQuestionWithDescriptionPromptBuilder,
            model_query_repeats=20 - count,
        )
        with open(f"forecasts/{question_id}.txt", "a") as file:
            file.writelines(f"{str(forecast)}\n" for forecast in forecasts)
