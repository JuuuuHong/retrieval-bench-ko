"""Ko-StrategyQA 로더 — MIRACL-ko와 성격이 다른 두 번째 한국어 검색 데이터셋.

`taeminlee/Ko-StrategyQA` (BEIR 포맷): StrategyQA(영문 다단계 추론 질문)를
하위 질문으로 분해해 한국어로 번역한 쿼리(qid 접미사 _0, _1 …) + 영문 위키
문단을 번역한 코퍼스 9,251건. dev 592쿼리, 쿼리당 정답 평균 1.9건.

MIRACL-ko와 다른 점(= 일반화 검증 축이 되는 이유):
- 질문·문단이 번역문이다 — 원문 한국어인 MIRACL과 문체가 다르다
- 원문이 영문 위키라, 한국어 위키 원문인 MIRACL과 주제 분포가 다르다
- 코퍼스가 9,251건으로 작다 — 정답 외 문서가 MIRACL보다 훨씬 적다
- multilingual-e5의 2단계 지도학습 데이터 목록(모델 카드)에 StrategyQA가 없다

코퍼스가 작아 서브샘플링 없이 전체를 쓴다. 무결성 처리는 MIRACL 로더와 같은
reconcile_qrels를 공유한다.
"""

from __future__ import annotations

import csv
import json

from retrieval_bench.data.miracl import RetrievalDataset, reconcile_qrels

_REPO = "taeminlee/Ko-StrategyQA"


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_REPO, filename, repo_type="dataset")


def parse_qrels_tsv(lines) -> dict[str, set[str]]:
    """BEIR qrels TSV(header: query-id, corpus-id, score) → qid -> 정답 docid 집합."""
    qrels: dict[str, set[str]] = {}
    for row in csv.DictReader(lines, delimiter="\t"):
        if float(row["score"]) > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def load_ko_strategyqa(split: str = "dev") -> RetrievalDataset:
    with open(_download(f"qrels/{split}.tsv"), encoding="utf-8") as f:
        qrels = parse_qrels_tsv(f)
    queries: dict[str, str] = {}
    with open(_download("queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]
    corpus: dict[str, str] = {}
    with open(_download("corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = f"{row.get('title') or ''} {row.get('text') or ''}".strip()

    qrels, queries, stats = reconcile_qrels(qrels, queries, corpus.keys())
    if stats["qrels_docs_missing_from_corpus"] or stats["queries_dropped"]:
        print(
            f"경고: 코퍼스에 없는 정답 문서 {stats['qrels_docs_missing_from_corpus']}건 "
            f"제외, 쿼리 {stats['queries_dropped']}건 드롭 "
            f"(평가 쿼리 {stats['queries_evaluated']}건)",
            flush=True,
        )
    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels, stats=stats)
