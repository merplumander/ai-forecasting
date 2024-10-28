# %%
import requests


def get_open_questions(page=1):
    url = f"https://www.metaculus.com/api2/questions/?status=open&page={page}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["results"]
    else:
        print("Error fetching open questions:", response.status_code)
        return []


# Example usage

# %%
open_questions = get_open_questions()

# %%
for question in open_questions:
    print(f"ID: {question['id']}, Title: {question['title']}")
# %%
