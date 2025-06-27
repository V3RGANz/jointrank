dataset="dl19"
for reranker in joint10 joint oracle topdown setwise tourrank
do
    for llm in gpt-4.1-mini
    do
        echo "$llm $reranker $dataset"
        python -m jointrank.evaluation.run_bench \
            --configs \
            conf/base.yaml \
            conf/datasets/${dataset}.yaml \
            conf/models/${reranker}.yaml \
            conf/llm/${llm}.yaml
    done
done


