# %%
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.dataset.dataset import (
    BinaryQuestion,
    QuestionJSONEncoder,
    question_json_decoder,
)
from src.dataset.metaculus_api import get_question_details, list_questions
from src.news.information_retrieval import search_web_and_summarize
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
from src.query.PromptBuilder import BinaryQuestionWithDescriptionAndNewsPromptBuilder
from src.utils import ROOT

# %%
load_dotenv(".env")

METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN")
TOURNAMENT_ID = 32506  # 32506 is the tournament ID for Q4 AI Benchmarking

save_folder = ROOT / "forecasts" / "metaculus-tournament" / "Q4"
# %%
questions = list_questions(
    tournament_id=TOURNAMENT_ID, metaculus_token=METACULUS_TOKEN, count=100
)

open_questions_ids = []
for question in questions["results"]:
    if question["status"] == "open":
        # print(
        #     f"ID: {question['id']}\nQ: {question['title']}\nCloses:"
        #     f" {question['scheduled_close_time']}"
        # )
        open_questions_ids.append(question["id"])
assert len(open_questions_ids) > 4, "Less than five open questions seems fishy."

# %%
for question_id in open_questions_ids:

    question_details = get_question_details(
        question_id, metaculus_token=METACULUS_TOKEN
    )

    title = question_details["question"]["title"]
    print(title)

# %%
for question_id in open_questions_ids:
    print(question_id)
    question_details = get_question_details(
        question_id, metaculus_token=METACULUS_TOKEN
    )
    assert question_details["question"]["type"] == "binary", "Question is not binary"
    date_string = question_details["created_at"]
    created_at = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
    resolved = question_details["status"] == "resolved"

    title = question_details["question"]["title"]
    resolution_criteria = question_details["question"]["resolution_criteria"]
    background = question_details["question"]["description"]
    fine_print = question_details["question"]["fine_print"]
    description = (
        f"Background: \n{background}\n\n"
        f"Fine Print: \n{fine_print}\n\n"
        f"Resolution Criteria: \n{resolution_criteria}\n\n"
    )
    question = BinaryQuestion(
        question_id=question_id,
        title=title,
        created_at=created_at,
        resolved=resolved,
        description=description,
    )

    news_summary = search_web_and_summarize(question, max_n_relevant_articles=10)

    question.news_summary = news_summary

    print(f"------------------------\nQuestion: {title}\n\n{description}")
    break


# %%
ensemble_models = [
    GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-flash-001"),
]

# ensemble_models = [
#     OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o"),
#     OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o-mini"),
#     AnthropicModel(
#         os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022"
#     ),
#     GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"),
#     GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-flash-001"),
#     LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.1-405b"),
#     LLAMAModel(os.environ.get("LLAMA_API_KEY"), "llama3.2-90b-vision"),
#     MistralModel(os.environ.get("MISTRAL_API_KEY"), "mistral-large-2407"),
#     MistralModel(os.environ.get("MISTRAL_API_KEY"), "mistral-small-2402"),
#     XAIModel(os.environ.get("XAI_API_KEY"), "grok-beta"),
#     QwenModel(os.environ.get("DASHSCOPE_API_KEY"), "qwen-max"),
#     QwenModel(os.environ.get("DASHSCOPE_API_KEY"), "qwen-plus"),
# ]
ensemble = ModelEnsemble(ensemble_models)

# # bqdsp: Binary Question with Description System Prompt
# # we have a number of different system prompts to increase diversity among the
# # predictions and to be able to do an evolutionary prompt improvement
# system_prompt_ids = [
#     "bqdsp_0",
#     "bqdsp_1",
#     "bqdsp_2",
#     "bqdsp_3",
#     "bqdsp_4",
#     "bqdsp_5",
#     "bqdsp_6",
#     "bqdsp_7",
#     "bqdsp_8",
#     "bqdsp_9",
#     "bqdsp_A",
#     "bqdsp_B",
# ]
system_prompt_ids = [
    "bqdsp_4",
]

forecasts = ensemble.make_forecast_from_question(
    question,
    BinaryQuestionWithDescriptionAndNewsPromptBuilder,
    system_prompt_ids=system_prompt_ids,
)


forecasts_save_path = Path(f"{save_folder}/{question_id}-forecasts.txt")
forecasts_save_path.parent.mkdir(exist_ok=True, parents=True)

with open(forecasts_save_path, "a") as file:
    file.writelines(f"{str(forecast)}\n" for forecast in forecasts)

# with open(f"{save_folder}/{question_id}-question.txt", "w") as file:
#     json.dump(question, file, cls=QuestionJSONEncoder, indent=4)

# %%
# with open(f"{save_folder}/{question_id}-question.txt", "r") as file:
#     serialize_str = file.read()
# deserialized_q = json.loads(serialize_str, object_hook=question_json_decoder)
#
#
#
#
# probability, comment = get_gpt_prediction(question_details)

# print(f"\n\n------------------LLM RESPONSE------------\n\n")
# print(f"--------------\nProbability: {probability}\n\nComment: {comment}\n\n")
# print(f"\n\n------------------END LLM RESPONSE------------\n\n")

# if SUBMIT_PREDICTION:
#     assert probability is not None, "Unexpected probability format"
#     post_question_prediction(question_id, probability)
#     post_question_comment(question_id, comment)

# %%
