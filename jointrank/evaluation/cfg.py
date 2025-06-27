from pathlib import Path
from typing import Literal
from pydantic import BaseModel
import yaml


class DatasetConfig(BaseModel):
    dataset: str
    pyserini_index: str | None = None
    run_path: str


class LLMConfig(BaseModel):
    engine: Literal["openai", "together"]
    model: str
    max_tokens: int | None = None


class RankerConfig(BaseModel):
    name: str
    block_size: int = 20  # a.k.a. window size (rankgpt), num children (setwise), etc.
    replicas: int = 2
    listwise: Literal["rankgpt", "custom"] = "custom"
    logits: bool = False
    num_tournaments: int = 1
    window_step: int = 10


class RunConfig(BaseModel):
    dataset: DatasetConfig
    llm: LLMConfig
    ranker: RankerConfig
    candidates_pool_size: int = 100
    shuffle_candidates: bool = False
    overwrite_results: bool = False


def _deep_update(d1: dict, d2: dict) -> dict:
    for k, v2 in d2.items():
        if isinstance(v2, dict) and isinstance((v1 := d1.get(k)), dict):
            d1[k] = _deep_update(v1, v2)
        else:
            d1[k] = v2
    return d1


def get_run_config(*dicts: dict) -> RunConfig:
    cfg_dict: dict = {}
    for d in dicts:
        _deep_update(cfg_dict, d)
    print(yaml.safe_dump(cfg_dict))  # noqa: T201
    return RunConfig.model_validate(cfg_dict)


def read_and_parse_configs(*paths: str) -> RunConfig:
    dicts = [yaml.safe_load(Path(p).expanduser().read_text()) for p in paths]
    return get_run_config(*dicts)
