# %%
import os
from pathlib import Path

from dotenv import load_dotenv

from src.dataset.metaculus_api import get_question_details, list_questions

# %%
load_dotenv(".env")

ROOT = Path(__file__).parent
METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN")
TOURNAMENT_ID = 32506  # 32506 is the tournament ID for Q4 AI Benchmarking
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

save_folder = ROOT / "forecasts" / "metaculus-tournament" / "Q4"
# %%
for question_id in open_questions_ids[0:2]:
    print(question_id)
    question_details = get_question_details(
        question_id, metaculus_token=METACULUS_TOKEN
    )

    title = question_details["question"]["title"]
    resolution_criteria = question_details["question"]["resolution_criteria"]
    background = question_details["question"]["description"]
    fine_print = question_details["question"]["fine_print"]

    # question = BinaryQuestion
    # print(
    #     f"------------------------\nQuestion: {title}\n\nResolution criteria:"
    #     f" {resolution_criteria}\n\nDescription: {background}\n\nFine print:"
    #     f" {fine_print}\n\n"
    # )

    # ensemble_models = [
    #     OpenAIModel(os.environ.get("OPENAI_API_KEY"), "gpt-4o"),
    #     AnthropicModel(
    #         os.environ.get("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022"
    #     ),
    #     GeminiModel(os.environ.get("GEMINI_API_KEY"), "gemini-1.5-pro-001"),
    # ]
    # ensemble = ModelEnsemble(ensemble_models)

    # # bqdsp: Binary Question with Description System Prompt
    # # we have a number of different system prompts to increase diversity amonng the
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
    # forecasts = ensemble.make_forecast_from_question(
    #     question,
    #     BinaryQuestionWithDescriptionAndNewsPromptBuilder,
    #     system_prompt_ids=system_prompt_ids,
    # )

    # with open(f"{save_folder}/{question_id}.txt", "a") as file:
    #     file.writelines(f"{str(forecast)}\n" for forecast in forecasts)
    #
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
