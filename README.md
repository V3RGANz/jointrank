# JointRank: Rank large set with single pass

## Benchmarks

Repository JointRank implementation, evaluation & experiments.

Benchmark evaluation can be run using `jointrank/evaluation/run_bench.py` script.
Configuration defined by `.yaml` config files. Presets for BEIR and TREC-2019 benchmarks already included and can be run with
prepared scripts.

### TREC eval

```shell
uv run scripts/prepare_trec.sh  # Downloading BM-25 index and obtaining top-1000
uv run scripts/rerank_trec.sh   # Rerank
RUN=path/to/run/file uv run scripts/dl19_eval.sh  # Evaluate
```

### BEIR eval

```shell
uv run scripts/prepare_beir.sh  # Downloading SPLADE++ index and obtaining top-1000
uv run scripts/rerank_beir.sh   # Rerank
uv run scripts/eval_beir.sh     # Evaluate
```

## Synthetic experiments

```shell
uv run python -m jointrank.evaluation.synthetic.coverage -k 10 -r 2 -v 100 -d latin
```

```shell
uv run python -m jointrank.evaluation.synthetic.connectivity -k 10 -r 2 -v 100 -d equi-replicate
```

```shell
uv run python -m jointrank.evaluation.synthetic.oracle --dst syn/oracle.csv
```
