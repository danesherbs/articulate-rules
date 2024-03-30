# %%

import time
from typing import Any
import openai

from pathlib import Path
from typeguard import typechecked

DATASETS_DIR = Path("datasets")
BLOBS_DIR = DATASETS_DIR / "blobs"

# %%


@typechecked
def get_model(job_id: str) -> str:
    """Get the model from fine-tuning job."""

    response = openai.FineTune.list()

    for job in response.data:
        if job.id != job_id:
            continue

        if job.fine_tuned_model is None:
            raise ValueError(f"Job ID {job_id} doesn't have a fine-tuned model.")

        return job.fine_tuned_model

    raise ValueError(f"Job ID {job_id} not found.")


# %%


@typechecked
def get_file(fpath: str | Path) -> Any:
    """Gets fine-tuning file at `fpath`. Uploads and returns it if it doesn't exist."""

    if not isinstance(fpath, Path):
        fpath = Path(fpath)

    if not fpath.is_file():
        raise FileNotFoundError(f"{fpath} doesn't exist.")

    response = openai.File.list()

    for file in response.data:
        if file.filename == fpath.name:
            return file

    response = openai.File.create(
        file=open(fpath, "rb"),
        purpose="fine-tune",
    )

    return response


# %%


@typechecked
def wait(job_id: str, interval: int = 60) -> None:
    """Blocks until fine-tuning job is complete."""

    while True:
        response = openai.FineTune.retrieve(job_id)

        print(f"Job '{job_id}' status: {response.status}.")

        if response.status in ["succeeded", "failed", "cancelled"]:
            break

        time.sleep(interval)


NOTABLE_FINETUNED_MODELS = {
    "davinci-fine-tuned-on-300-in-distribution-classification-tasks": (
        "davinci:ft-honest-articulation-project-2023-06-14-07-39-40"
    ),
    "davinci-fine-tuned-on-300-in-distribution-classification-tasks-and-500-out-of-distribution-classification-tasks": (
        "davinci:ft-honest-articulation-project-2023-06-16-05-08-57"
    ),
}
