"""검색 품질 메트릭.

모든 함수는 단일 쿼리 기준으로 계산하고, aggregate()가 쿼리 평균을 낸다.
- ranked: 검색기가 반환한 문서 ID 리스트 (점수 내림차순)
- relevant: 정답(qrels) 문서 ID 집합
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """정답 문서 중 상위 k개 안에 들어온 비율."""
    if not relevant:
        raise ValueError("정답 집합이 비어 있으면 recall을 정의할 수 없습니다")
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """첫 번째 정답 문서 순위의 역수. 상위 k개 안에 정답이 없으면 0."""
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """이진 관련성 기준 nDCG. 정답이면 gain=1, 아니면 0."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(ranked[:k])
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_metrics(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    """쿼리별 메트릭 — 신뢰구간·유의성 검정·오류 분석의 원재료.

    집계값만 남기고 per-query를 버리면 사후 검정이 재실행 없이는 불가능해진다.
    """
    qids = [qid for qid in runs if qrels.get(qid)]
    if not qids:
        raise ValueError("평가 가능한 쿼리가 없습니다 (qrels 매칭 0건)")

    # 중복 문서 ID는 nDCG를 1 초과로 부풀린다 — 검색기 버그의 신호이므로
    # 무음으로 계산하지 않고 즉시 실패시킨다
    for qid in qids:
        if len(set(runs[qid])) != len(runs[qid]):
            raise ValueError(f"검색 결과에 중복 문서 ID: qid={qid}")

    out: dict[str, dict[str, float]] = {}
    for qid in qids:
        m: dict[str, float] = {}
        for k in k_values:
            m[f"recall@{k}"] = recall_at_k(runs[qid], qrels[qid], k)
            m[f"mrr@{k}"] = mrr_at_k(runs[qid], qrels[qid], k)
            m[f"ndcg@{k}"] = ndcg_at_k(runs[qid], qrels[qid], k)
        out[qid] = m
    return out


def aggregate(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k_values: list[int],
) -> dict[str, float]:
    """전체 쿼리에 대한 메트릭 평균.

    runs: qid -> 검색 결과 문서 ID 리스트
    qrels: qid -> 정답 문서 ID 집합
    반환: {"recall@10": 0.83, "mrr@10": ..., "ndcg@10": ...}
    """
    per_query = per_query_metrics(runs, qrels, k_values)
    qids = list(per_query)
    result: dict[str, float] = {}
    for k in k_values:
        for name in (f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"):
            result[name] = sum(per_query[q][name] for q in qids) / len(qids)
    result["num_queries"] = len(qids)
    return result
