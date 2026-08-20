from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def data_path(config: dict, *parts: str) -> Path:
    path = Path(config["data_dir"])
    return path.joinpath(*parts)
