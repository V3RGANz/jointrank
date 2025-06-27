python -m pyserini.search.lucene \
  --threads 16 \
  --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output data/run.msmarco-v1-passage.bm25-default.dl19.txt \
  --bm25 --k1 0.9 --b 0.4
