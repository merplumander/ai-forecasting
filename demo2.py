import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
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
from src.query.utils import aggregate_forecasting_explanations
from src.utils import ROOT

load_dotenv(".env")
ENSEMBLE = ModelEnsemble(
    [
        OpenAIModel(os.environ.get("OPENAI_API_KEY")),
        AnthropicModel(os.environ.get("ANTHROPIC_API_KEY")),
        GeminiModel(os.environ.get("GEMINI_API_KEY")),
        # XAIModel(os.environ.get("XAI_API_KEY")),
        LLAMAModel(os.environ.get("LLAMA_API_KEY")),
        # MistralModel(os.environ.get("MISTRAL_API_KEY")),
        QwenModel(os.environ.get("DASHSCOPE_API_KEY")),
    ]
)
with open(
    ROOT / "prompts" / "binary_question_with_description_system_prompt.txt", "r"
) as file:
    # Read the entire content of the file
    CONTEXT = file.read()


def ask_question(question):
    responses = ENSEMBLE.make_forecast(
        question,
        CONTEXT,
    )
    probabilities = [response[1] for response in responses if response[1] is not None]
    models = [response[0] for response in responses if response[0] is not None]
    lower, median, upper = np.quantile(probabilities, [0.05, 0.50, 0.95])
    language_model = OpenAIModel(
        api_key=os.environ.get("OPENAI_API_KEY"), model_version="gpt-4o"
    )
    reasonings = [response[1] for response in responses]
    aggregated_explanation = aggregate_forecasting_explanations(
        question=question,
        reasonings=reasonings,
        probabilities=probabilities,
        language_model=language_model,
    )
    fig = plot_results(probabilities, models)

    return (
        f"Median: {round(median, 2)} %",
        f"90% Confidence Interval: {round(lower, 2)}% - {round(upper, 2)}%",
        aggregated_explanation,
        fig,
    )


with gr.Blocks() as demo:
    gr.Markdown("# Ask a Question")
    question = gr.Textbox(label="Type your question here")
    submit_btn = gr.Button("Submit")
    median_output = gr.Textbox(label="Median")
    confidence_output = gr.Textbox(label="Confidence Interval")
    explanation_output = gr.Textbox(label="Explanation")

    def plot_results(probabilities, models):
        fig, ax = plt.subplots()
        sns.boxplot(x=probabilities, showfliers=False)
        sns.swarmplot(x=probabilities, hue=models, ax=ax, size=8)
        ax.legend(title="Models")
        ax.set_xlabel("Probability")
        ax.set_title("Model Forecast Probabilities")
        ax.set_xlim(0, 100)
        return fig

    plot_output = gr.Plot(label="Results Plot")

    submit_btn.click(
        ask_question,
        inputs=question,
        outputs=[median_output, confidence_output, explanation_output, plot_output],
    )

demo.launch()
