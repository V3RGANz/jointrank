# Metrics calculation for existing runs

dataset="dl19"

echo "BM25"
python -m pyserini.eval.trec_eval \
  -c -l 2 -m ndcg_cut.10 dl19-passage \
  data/run.msmarco-v1-passage.bm25-default.dl19.txt \
  | awk 'NR==1{print $NF}'
echo ""

for run in ./runs/msmarco-passage/trec-dl-2019/*; do
    if [[ -f "$run" && "$(basename "$run")" != *.yaml ]]; then
        echo "$(awk 'NR==1{print $NF}' "$run") $(basename "$run")"
        python -m pyserini.eval.trec_eval \
            -c -l 2 -m ndcg_cut.10 dl19-passage \
            $run | awk 'NR==1{print $NF}'
        echo ""
    fi
done
