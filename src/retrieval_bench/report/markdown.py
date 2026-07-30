"""실험 결과를 README에 붙여넣을 수 있는 마크다운 표로 변환."""

from __future__ import annotations


def metrics_table(experiment: str, metrics: dict[str, float]) -> str:
    keys = [k for k in metrics if k != "num_queries"]
    header = "| experiment | " + " | ".join(keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    row = (
        f"| {experiment} | "
        + " | ".join(f"{metrics[k]:.4f}" for k in keys)
        + " |"
    )
    return "\n".join([header, sep, row])


# README 비교표에 싣는 열 — 전체 메트릭 중 의사결정에 쓰는 것만
DEFAULT_COLUMNS = [
    "recall@10",
    "recall@100",
    "ndcg@10",
    "tokens@10",
    "latency_p50_ms",
    "latency_p95_ms",
]


def combined_table(
    results: list[dict],
    columns: list[str] | None = None,
) -> str:
    """여러 실험 결과 JSON(dict)을 하나의 비교표로.

    각 result는 러너가 저장하는 형태({"experiment", "corpus_size", "metrics"}).
    열에 없는 메트릭은 '-'로 표시. 행 순서는 입력 순서를 따른다.
    """
    if not results:
        raise ValueError("결과가 없습니다")
    columns = columns or DEFAULT_COLUMNS
    lines = [
        "| experiment | corpus | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 2) + "|",
    ]
    for r in results:
        m = r["metrics"]
        cells = [
            f"{m[c]:.4f}" if isinstance(m.get(c), float) else str(m.get(c, "-"))
            for c in columns
        ]
        lines.append(
            f"| {r['experiment']} | {r.get('corpus_size', '-'):,} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)
