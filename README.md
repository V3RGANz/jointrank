# JointRank: Rank large set with single pass

Repository JointRank implementation, evaluation & experiments.

Paper (ACM free access): https://doi.org/10.1145/3731120.3744587

[arXiv preprint](https://arxiv.org/abs/2506.22262)


## Project setup

I use uv for package/project management, so before environment setup, please [install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

> You can also use any other (your favorite) package manager, in this case, manually install dependencies listed in `pyproject.toml`.

Install requirements with uv:

```shell
uv sync
```

## Benchmarks

Benchmark evaluation can be run using `jointrank/evaluation/run_bench.py` script.
Configuration defined by `.yaml` config files. Presets for BEIR and TREC-2019 benchmarks already included and can be run with
prepared scripts.

### TREC eval

```shell
uv run scripts/prepare_trec.sh  # Downloading BM-25 index and obtaining top-1000
uv run scripts/rerank_trec.sh   # Rerank top-100
uv run scripts/eval_dl19.sh     # Evaluate
```

### BEIR eval

```shell
uv run scripts/prepare_beir.sh  # Downloading BM-25 index and obtaining top-1000
uv run scripts/rerank_beir.sh   # Rerank top-100
uv run scripts/eval_beir.sh     # Evaluate
```

## Synthetic experiments

```shell
uv run python -m jointrank.evaluation.synthetic.oracle --dst syn/oracle.csv
```

```shell
uv run python -m jointrank.evaluation.synthetic.coverage -k 10 -r 2 -v 100 -d latin
```

## Shuffle and rerank top-1000

See `conf/top1000shuffle.yaml` and `conf/models1000/*.yaml` configs and use them instead of `conf/base.yaml` and `conf/models/*.yaml` in script `rerank_trec.sh`.

# Citation

```bib
@inproceedings{10.1145/3731120.3744587,
author = {Dedov, Evgeny},
title = {JointRank: Rank Large Set with Single Pass},
year = {2025},
isbn = {9798400718618},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3731120.3744587},
doi = {10.1145/3731120.3744587},
abstract = {Efficiently ranking relevant items from large candidate pools is a cornerstone of modern information retrieval systems - such as web search, recommendation, and retrieval-augmented generation. Listwise rerankers, which improve relevance by jointly considering multiple candidates, are often limited in practice: either by model input size constraints, or by degraded quality when processing large sets. We propose a model-agnostic method for fast reranking large sets that exceed a model input limits. The method first partitions candidate items into overlapping blocks, each of which is ranked independently in parallel. Implicit pairwise comparisons are then derived from these local rankings. Finally, these comparisons are aggregated to construct a global ranking using algorithms such as Winrate or PageRank. Experiments on TREC DL-2019 show that our method achieves an nDCG@10 of 70.88 compared to the 57.68 for full-context listwise approach using gpt-4.1-mini as long-context model, while reducing latency from 21 to 8 seconds. The implementation of the algorithm and the experiments is available in the repository: https://github.com/V3RGANz/jointrank},
booktitle = {Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR)},
pages = {208–217},
numpages = {10},
keywords = {block design, large language models for zero-shot ranking},
location = {Padua, Italy},
series = {ICTIR '25}
}
```
