# Metrics calculation for existing runs

for dataset in trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus
do
    echo "=========="
    echo "${dataset}"
    echo "=========="

    echo "BM25"
    python -m pyserini.eval.trec_eval \
      -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
      data/run.beir.bm25-flat.${dataset}.txt \
      | awk 'NR==1{print $NF}'
    echo ""

    for run in ./runs/beir/$dataset/*; do
        if [[ -f "$run" && "$(basename "$run")" != *.yaml ]]; then
            echo "$(awk 'NR==1{print $NF}' "$run") $(basename "$run")"
            python -m pyserini.eval.trec_eval \
                -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
                $run | awk 'NR==1{print $NF}'
            echo ""
        fi
    done
done
