# %%
import os
import re

import numpy as np
import openai
from dotenv import load_dotenv


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
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
# load context prompt
with open("context_prompt.txt", "r") as file:
    # Read the entire content of the file
    context_prompt = file.read()


# %%
forecasting_question = (
    "Will global average temperatures be more than 2 degrees Celsius above"
    " pre-industrial levels by the year 2050?"
)

# %%
single_model_repetitions = 3
gpt_responses = []

for i in range(single_model_repetitions):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{forecasting_question}"},
            {"role": "system", "content": f"{context_prompt}"},
        ],
        max_tokens=100,
        temperature=2,
        top_p=0.8,
    )

    reply = response.choices[0].message.content
    # print("ChatGPT's response:", reply)
    try:
        number = validate_asterisk_number(reply)
        gpt_responses.append(number)
    except ValueError as e:
        print(
            f"Validation error: {e}   Nothing added to gpt_responses. ChatGPT's"
            f" response was: {reply}"
        )
# %%
# more complicated but perhaps useful once we weight different forecasters
# import scipy.stats as st
# rv = st.rv_discrete(values=(values, weights/weights.sum()))
# print("median:", rv.median())
# print("68% CI:", rv.interval(0.90))

lower, median, upper = np.quantile(gpt_responses, [0.10, 0.50, 0.90])
print(
    f"The predicted probability is: {median:.0f}%. With a 90% confidence interval of"
    f" [{lower:.0f}%, {upper:.0f}%]"
)
# %%
