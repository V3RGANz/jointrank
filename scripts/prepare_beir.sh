# prepare BM25 first-stage results
# https://castorini.github.io/pyserini/2cr/beir.html

set -x

for dataset in trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus
do
    python -m pyserini.search.lucene \
      --threads 16 --batch-size 128 \
      --index beir-v1.0.0-${dataset}.flat \
      --topics beir-v1.0.0-${dataset}-test \
      --output data/run.beir.bm25-flat.${dataset}.txt \
      --output-format trec \
      --hits 1000 --bm25 --remove-query
done
