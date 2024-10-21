# %%
import openai
import os
from dotenv import load_dotenv

# %%
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
# Send a prompt to the ChatGPT API
# response = openai.ChatCompletion.create(
#     model="gpt-4",  # or "gpt-3.5-turbo" for a less expensive option
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "How do I send text to ChatGPT via the API using the latest openai package?"},
#     ],
#     max_tokens=150,  # Maximum number of tokens to generate in the response
#     temperature=0.7  # Controls the randomness of the response
# )

# prompt = input("Please enter a question or request: ")

# %%
response = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "Who are you?"
        }
    ]
)

# %%
# Extract the response text
reply = response.choices[0].message.content
print("ChatGPT's response:", reply)
# %%
