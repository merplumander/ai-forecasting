import os

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.query.language_models import (
    AnthropicModel,
    GeminiModel,
    LLAMAModel,
    OpenAIModel,
    XAIModel,
)
from src.query.ModelEnsemble import ModelEnsemble
from src.query.utils import aggregate_forecasting_explanations

app = Flask(__name__, template_folder="src/demo")

load_dotenv(".env")
ENSEMBLE = ModelEnsemble(
    [
        OpenAIModel(os.environ.get("OPENAI_API_KEY")),
        AnthropicModel(os.environ.get("ANTHROPIC_API_KEY")),
        GeminiModel(os.environ.get("GEMINI_API_KEY")),
        # XAIModel(os.environ.get("XAI_API_KEY")),
        LLAMAModel(os.environ.get("LLAMA_API_KEY")),
    ]
)
with open("context_prompt_logprobs.txt", "r") as file:
    # Read the entire content of the file
    CONTEXT = file.read()


# Route to render the HTML form
@app.route("/")
def home():
    return render_template("index.html")


# Route to handle the question submission
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    responses = ENSEMBLE.make_forecast(
        question,
        CONTEXT,
    )
    probabilities = [response[1] for response in responses if response[1] is not None]
    print(probabilities)
    lower, median, upper = np.quantile(probabilities, [0.05, 0.50, 0.95])
    # closest_response_idx = np.argmin(np.abs(median - probabilities))
    # explanation = responses[1][closest_response_idx]
    language_model = OpenAIModel(
        api_key=os.environ.get("OPENAI_API_KEY"), model_version="gpt-4o"
    )
    aggregated_explanation = aggregate_forecasting_explanations(
        question=question,
        reasonings=responses[1],
        probabilities=probabilities,
        language_model=language_model,
    )
    return jsonify(
        {
            "median": round(median, 2),
            "confidence_interval": (round(lower, 2), round(upper, 2)),
            "explanation": aggregated_explanation,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
