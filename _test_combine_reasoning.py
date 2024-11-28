# %%

question = "will bla be blu?"
reasonings = ["bla", "blabla"]
median_forecast = 80
confidence_range_string = "[60%, 90%]"
newline = "\n"
reasonings = f"{newline.join(
    f"Reasoning {number}:\n{reasoning}\n"
    for number, reasoning in enumerate(reasonings))}"
with open("prompts/combine_reasoning_prompt.txt", "r") as file:
    # Read the entire content of the file
    combine_prompt = file.read()

combine_prompt = combine_prompt.format(
    question=question,
    reasonings=reasonings,
    median_forecast=median_forecast,
    confidence_range_string=confidence_range_string,
)
# %%
print(combine_prompt)
# %%
