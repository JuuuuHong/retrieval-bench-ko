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
