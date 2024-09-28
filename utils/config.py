import yaml
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)
