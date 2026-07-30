"""MIRACL-ko 로더.

원본 `miracl/miracl`은 로딩 스크립트 기반이라 datasets 3.x+에서 로드 불가
(스크립트 지원 제거). 대신 BEIR 포맷 표준 Parquet 미러인
`taeminlee/Ko-miracl`을 사용한다 (gated 아님, 로그인 불필요).

- corpus  (config=corpus,  split=corpus): _id / title / text — 1,486,752건
- queries (config=queries, split=queries): _id / text
- qrels   (config=default, split=train|dev): query-id / corpus-id / score
  score>0 이 정답. score=0(판정된 비관련 문서)도 하드 네거티브로 코퍼스 풀에 포함.

코퍼스 구성 전략 (1주차):
- 기본 풀: qrels에 등장하는 모든 문서 (정답 + 하드 네거티브)
- corpus_sample 지정 시: 전체 코퍼스에서 시드 고정 무작위 문서를 추가
  (전체 148만 건은 rank_bm25로 비현실적이라 서브셋으로 측정하고,
   모든 결과표에 코퍼스 크기를 명시한다)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_REPO = "taeminlee/Ko-miracl"


@dataclass
class RetrievalDataset:
    queries: dict[str, str]        # qid -> 쿼리 텍스트
    corpus: dict[str, str]         # docid -> "제목 본문"
    qrels: dict[str, set[str]]     # qid -> 정답 docid 집합


def _doc_text(row: dict) -> str:
    title = row.get("title") or ""
    return f"{title} {row['text']}".strip()


def load_miracl_ko(
    split: str = "dev",
    corpus_sample: int = 0,
    seed: int = 42,
) -> RetrievalDataset:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets 가 필요합니다") from e

    qrels_ds = load_dataset(_REPO, "default", split=split)
    queries_ds = load_dataset(_REPO, "queries", split="queries")
    corpus_ds = load_dataset(_REPO, "corpus", split="corpus")

    # qrels: 정답(score>0)과 판정 문서 전체(하드 네거티브 포함)를 분리 수집
    qrels: dict[str, set[str]] = {}
    judged_docids: set[str] = set()
    for row in qrels_ds:
        qid, docid = str(row["query-id"]), str(row["corpus-id"])
        judged_docids.add(docid)
        if float(row["score"]) > 0:
            qrels.setdefault(qid, set()).add(docid)

    # 쿼리: 해당 스플릿 qrels에 등장하는 것만
    queries = {
        str(row["_id"]): row["text"]
        for row in queries_ds
        if str(row["_id"]) in qrels
    }

    # 코퍼스 풀: 판정 문서 + 무작위 distractor
    # 주의: set 순회는 해시 랜덤화로 프로세스마다 순서가 달라져 동점 문서의
    # 정렬 순서(argsort 안정성)가 흔들린다 → 정렬해서 코퍼스 순서를 고정
    corpus: dict[str, str] = {}
    docid_to_idx = {str(d): i for i, d in enumerate(corpus_ds["_id"])}
    for docid in sorted(judged_docids):
        idx = docid_to_idx.get(docid)
        if idx is not None:
            corpus[docid] = _doc_text(corpus_ds[idx])

    if corpus_sample > len(corpus):
        rng = random.Random(seed)
        need = corpus_sample - len(corpus)
        # 전체 인덱스에서 시드 고정 샘플링 (이미 포함된 문서는 건너뜀)
        candidates = rng.sample(range(len(corpus_ds)), min(need * 2, len(corpus_ds)))
        for idx in candidates:
            if len(corpus) >= corpus_sample:
                break
            row = corpus_ds[idx]
            docid = str(row["_id"])
            if docid not in corpus:
                corpus[docid] = _doc_text(row)

    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels)
