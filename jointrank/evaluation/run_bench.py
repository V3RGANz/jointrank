import argparse
import logging
import random
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncClient
from tqdm.auto import tqdm

from jointrank.evaluation.cfg import (
    LLMConfig,
    RankerConfig,
    RunConfig,
    read_and_parse_configs,
)
from jointrank.evaluation.data import TRECDevRerankerSample, prepare_dataset
from jointrank.evaluation.reranker_factory import get_reranker
from jointrank.model.llm.openai import OpenAILLMEngine
from jointrank.model.llm.telemetry import LLMEngineWithTelemetry, Telemetry
from jointrank.model.llm.together import TogetherEngine
from jointrank.model.reranker_base import Reranker

LOG = logging.getLogger(__name__)


def get_output_path(cfg: RunConfig, parent: str = "results") -> str:
    def serialize_ranker_params(r_cfg: RankerConfig) -> str:
        params = ""
        if r_cfg.name in ["joint", "topdown", "sliding", "listwise"]:
            params += "-logits" if r_cfg.logits else ""
            if r_cfg.listwise != "listwise":
                params += f"-{r_cfg.listwise}"
        if r_cfg.name in ["joint", "setwise", "topdown", "sliding"]:
            params += f"-{r_cfg.block_size}"
        if r_cfg.name == "joint":
            params += f"-{r_cfg.replicas}"
        if r_cfg.name == "tourrank":
            params += f"-{r_cfg.num_tournaments}"
        if r_cfg.name == "sliding":
            params += f"-{r_cfg.window_step}"
        return params
    path = f"{cfg.ranker.name}-{cfg.candidates_pool_size}"
    if cfg.shuffle_candidates:
        path += "-s"
    path += serialize_ranker_params(cfg.ranker)
    if cfg.ranker.name != "oracle":
        path += f"-{cfg.llm.model.lower().replace('/', '-')}"
    return str(Path(parent) / cfg.dataset.dataset / path)


def get_llm(llm_config: LLMConfig) -> LLMEngineWithTelemetry:
    if llm_config.engine == "openai":
        llm = OpenAILLMEngine(AsyncClient(), llm_config.model)
    elif llm_config.engine == "together":
        llm = TogetherEngine(model=llm_config.model, max_tokens=llm_config.max_tokens)
    else:
        raise ValueError(llm_config.engine)
    return LLMEngineWithTelemetry(llm)


def run_reranker(
    reranker: Reranker, samples: list[TRECDevRerankerSample], telemetry: Telemetry, config: RunConfig
) -> list[tuple]:
    # 19335 Q0 8412684 1 10.606700 Anserini
    results = []
    for sample in tqdm(samples, desc="Evaluation"):
        documents = sample.documents[:config.candidates_pool_size]
        if config.shuffle_candidates:
            random.shuffle(documents)
        start = time.time()
        permutation = reranker.rerank(sample.query.content, [doc.as_str() for doc in documents])
        end = time.time()
        # include filtered out docs in the end for fare evaluation
        permutation_tail = [i for i in range(len(documents)) if i not in permutation]
        telemetry.total_time += end - start
        for rank, i in enumerate(permutation + permutation_tail, start=1):
            results.append((sample.query.idx, "Q0", documents[i].idx, rank, -rank, reranker.__class__.__name__))
    return results

# def run_trec_eval(
#     reranker: Reranker, samples: list[TRECDevRerankerSample], telemetry: Telemetry, args: argparse.Namespace
# ) -> list[tuple]:
#     # 19335 Q0 8412684 1 10.606700 Anserini
#     results = []
#     for sample in tqdm(samples, desc="Evaluation"):
#         documents = sample.documents[:args.candidates_pool_size]
#         if args.shuffle_candidates:
#             random.shuffle(documents)
#         start = time.time()
#         permutation = reranker.rerank(sample.query.content, [doc.as_str() for doc in documents])
#         end = time.time()
#         # include filtered out docs in the end for fare evaluation
#         permutation_tail = [i for i in range(len(documents)) if i not in permutation]
#         telemetry.total_time += end - start
#         # LOG.info(telemetry)
#         for rank, i in enumerate(permutation + permutation_tail, start=1):
#             results.append((sample.query.idx, "Q0", documents[i].idx, rank, -rank, reranker.__class__.__name__))
#     return results
#

def serialize_config_and_telemetry(cfg: RunConfig, telemetry: Telemetry, total_samples: int) -> dict:
    res = {}
    res["config"] = cfg.model_dump()

    res["telemetry"] = {
        "avg_input_tokens": telemetry.input_tokens / total_samples,
        "avg_output_tokens": telemetry.output_tokens / total_samples,
        "avg_inference_time": telemetry.inference_total_time / total_samples,
        "avg_total_time": telemetry.total_time / total_samples,
        "avg_inferences": telemetry.inferences / total_samples
    }

    LOG.info("input tokens %.2f", res["telemetry"]["avg_input_tokens"])
    LOG.info("output tokens %.2f", res["telemetry"]["avg_output_tokens"])
    LOG.info("inference time %.2f", res["telemetry"]["avg_inference_time"])
    LOG.info("total time %.2f", res["telemetry"]["avg_total_time"])
    LOG.info("inferences count %.2f", res["telemetry"]["avg_inferences"])

    return res


def main(cfg: RunConfig) -> None:
    results_path = get_output_path(cfg, "runs")
    if not cfg.overwrite_results and Path(results_path + ".txt").is_file():
        LOG.info("metrics already calculated at %s, nothing to do", results_path)
        return

    to_rerank, gold = prepare_dataset(cfg.dataset)
    llm = get_llm(cfg.llm)
    reranker = get_reranker(llm, cfg, to_rerank, gold)

    reranker_result = run_reranker(reranker, to_rerank, llm.telemetry, cfg)
    results_content = "\n".join(" ".join(map(str, r)) for r in reranker_result)
    run_path = results_path + ".txt"
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    Path(run_path).write_text(results_content)
    LOG.info("written results to %s", run_path)

    run_meta = serialize_config_and_telemetry(cfg, llm.telemetry, len(to_rerank))
    meta_path = Path(results_path + ".meta.yaml")
    meta_path.write_text(yaml.safe_dump(run_meta))
    LOG.info("written meta to %s", str(meta_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args()
    cfg = read_and_parse_configs(*args.configs)
    load_dotenv(override=True)

    main(cfg)
