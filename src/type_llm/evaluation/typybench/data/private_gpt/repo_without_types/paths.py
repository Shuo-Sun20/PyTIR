from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import settings


def _absolute_or_from_project_root(path):
    if path.startswith("/"):
        return Path(path)
    return PROJECT_ROOT_PATH / path


models_path = PROJECT_ROOT_PATH / "models"
models_cache_path = models_path / "cache"
docs_path = PROJECT_ROOT_PATH / "docs"
local_data_path = _absolute_or_from_project_root(
    settings().data.local_data_folder
)
