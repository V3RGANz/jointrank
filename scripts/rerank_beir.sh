# for dataset in trec-covid webis-touche2020 signal1m trec-news
for dataset in trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus
do
    for reranker in joint10 joint topdown sliding
    do
        llm="gpt-4.1-mini"
        # llm="mistral7b"
        python -m jointrank.evaluation.run_bench \
            --configs \
            conf/base.yaml \
            conf/datasets/beir-${dataset}.yaml \
            conf/models/${reranker}.yaml \
            conf/llm/${llm}.yaml
    done
done
