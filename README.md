# JointRank: Rank large set with single pass

Repository JointRank implementation, evaluation & experiments.

arXiv preprint: https://arxiv.org/abs/2506.22262

Version of Record: https://doi.org/10.1145/3731120.3744587 (to be published soon)

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

BibTeX reference will be specified soon, after official ACM publication.
