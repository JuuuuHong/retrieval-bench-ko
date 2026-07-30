"""MIRACL-ko 로더.

원본 `miracl/miracl`은 로딩 스크립트 기반이라 datasets 3.x+에서 로드 불가
(스크립트 지원 제거). 대신 BEIR 포맷 표준 Parquet 미러인
`taeminlee/Ko-miracl`을 사용한다 (gated 아님, 로그인 불필요).

- corpus  (config=corpus,  split=corpus): _id / title / text — 1,486,752건
- queries (config=queries, split=queries): _id / text
- qrels   (config=default, split=train|dev): query-id / corpus-id / score
  score>0 이 정답. score=0(판정된 비관련 문서)도 하드 네거티브로 코퍼스 풀에 포함.

코퍼스 구성 전략:
- 기본 풀: qrels에 등장하는 모든 문서 (정답 + 하드 네거티브)
- corpus_sample 지정 시: 전체 코퍼스에서 시드 고정 무작위 문서를 추가
  (전체 148만 건은 rank_bm25로 비현실적이라 서브셋으로 측정하고,
   모든 결과표에 코퍼스 크기를 명시한다)

무결성 원칙: 데이터 정합성 문제(정답 문서가 코퍼스에 없음, 쿼리 텍스트 없는
qrels, 코퍼스 목표 미달)는 절대 무음으로 넘기지 않는다 — 드롭하면 경고를
출력하고 stats에 기록해 결과 JSON까지 전파하며, 목표 미달은 예외로 중단한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_REPO = "taeminlee/Ko-miracl"


@dataclass
class RetrievalDataset:
    queries: dict[str, str]        # qid -> 쿼리 텍스트
    corpus: dict[str, str]         # docid -> "제목 본문"
    qrels: dict[str, set[str]]     # qid -> 정답 docid 집합
    stats: dict = field(default_factory=dict)  # 드롭 건수 등 무결성 기록


def _doc_text(row: dict) -> str:
    title = row.get("title") or ""
    text = row.get("text") or ""
    return f"{title} {text}".strip()


def select_corpus_docids(
    judged_docids: set[str],
    all_docids: list[str],
    corpus_sample: int,
    seed: int,
) -> list[str]:
    """서브셋에 포함할 docid를 결정론적으로 선택.

    판정 문서(정렬 순) 먼저, 그 뒤 시드 고정 셔플 순서의 무작위 distractor.
    - corpus_sample <= 0: 판정 문서만
    - 판정 문서가 corpus_sample보다 많으면 에러 (정답 절삭은 평가를 왜곡)
    - 목표 미달이면 에러 (조용히 다른 크기로 평가되는 것 방지)
    """
    available = set(all_docids)
    selected = [d for d in sorted(judged_docids) if d in available]
    if corpus_sample <= 0:
        return selected
    if len(selected) > corpus_sample:
        raise ValueError(
            f"판정 문서 {len(selected)}건 > corpus_sample {corpus_sample} — "
            "corpus_sample을 키우세요 (정답 절삭은 지원하지 않음)"
        )
    if corpus_sample > len(available):
        raise ValueError(
            f"corpus_sample {corpus_sample} > 전체 코퍼스 {len(available)}건"
        )
    chosen = set(selected)
    rng = random.Random(seed)
    order = list(range(len(all_docids)))
    rng.shuffle(order)
    for idx in order:
        if len(selected) >= corpus_sample:
            break
        docid = all_docids[idx]
        if docid not in chosen:
            chosen.add(docid)
            selected.append(docid)
    if len(selected) != corpus_sample:
        raise ValueError(
            f"코퍼스 목표 {corpus_sample}건 미달: {len(selected)}건"
        )
    return selected


def reconcile_qrels(
    qrels: dict[str, set[str]],
    queries: dict[str, str],
    corpus_docids: set[str],
) -> tuple[dict[str, set[str]], dict[str, str], dict]:
    """qrels·queries·corpus 3자 정합성을 맞추고 드롭 내역을 반환.

    - 코퍼스에 없는 정답 docid는 recall 분모에서 제외 (남겨두면 어떤 검색기도
      도달 불가능한 문서가 분모에 남아 모든 recall이 무음으로 왜곡된다)
    - 정답이 전멸한 쿼리, 쿼리 텍스트가 없는 qrels는 제거
    """
    all_relevant = set().union(*qrels.values()) if qrels else set()
    missing_docs = all_relevant - corpus_docids

    pruned: dict[str, set[str]] = {}
    dropped_queries: list[str] = []
    for qid, rel in qrels.items():
        kept = rel - missing_docs
        if not kept:
            dropped_queries.append(qid)
        elif qid not in queries:
            dropped_queries.append(qid)
        else:
            pruned[qid] = kept

    kept_queries = {q: t for q, t in queries.items() if q in pruned}
    stats = {
        "qrels_docs_missing_from_corpus": len(missing_docs),
        "queries_dropped": len(dropped_queries),
        "queries_evaluated": len(pruned),
    }
    return pruned, kept_queries, stats


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

    queries = {
        str(row["_id"]): row["text"]
        for row in queries_ds
        if str(row["_id"]) in qrels
    }

    all_docids = [str(d) for d in corpus_ds["_id"]]
    selected = select_corpus_docids(judged_docids, all_docids, corpus_sample, seed)
    docid_to_idx = {d: i for i, d in enumerate(all_docids)}
    corpus = {d: _doc_text(corpus_ds[docid_to_idx[d]]) for d in selected}

    qrels, queries, stats = reconcile_qrels(qrels, queries, corpus.keys())
    if stats["qrels_docs_missing_from_corpus"] or stats["queries_dropped"]:
        print(
            f"경고: 코퍼스에 없는 정답 문서 {stats['qrels_docs_missing_from_corpus']}건 "
            f"제외, 쿼리 {stats['queries_dropped']}건 드롭 "
            f"(평가 쿼리 {stats['queries_evaluated']}건)",
            flush=True,
        )

    return RetrievalDataset(queries=queries, corpus=corpus, qrels=qrels, stats=stats)
