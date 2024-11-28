# %%
from datetime import datetime

from src.dataset.dataset import MetaculusDataset

# %%

dataset = MetaculusDataset(
    path="ai-forecasting-datasets", file_infix="full", download_new_data=True
)
# %%
dataset.select_questions_with_status("resolved")
dataset.select_questions_with_forecast_type("binary")
dataset.select_questions_newer_than(datetime(year=2024, month=10, day=1))
dataset.file_infix = "resolved_binary_from_2024_09_01"
# dataset.save()
# %%
dataset = MetaculusDataset(
    path="ai-forecasting-datasets",
    file_infix="resolved_binary_from_2024_01_01",
    download_new_data=False,
)
# %

# %%
