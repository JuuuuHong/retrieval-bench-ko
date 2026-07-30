"""두 실험 결과의 유의성 비교.

사용법:
    python -m retrieval_bench.compare results/dense_e5_small.json results/hybrid_rrf.json recall@10

결과 JSON의 per_query 필드(run.py가 저장)를 사용한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrieval_bench.eval.significance import paired_bootstrap


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        raise SystemExit(1)
    path_a, path_b, metric = sys.argv[1], sys.argv[2], sys.argv[3]
    ra = json.loads(Path(path_a).read_text())
    rb = json.loads(Path(path_b).read_text())
    pa, pb = ra.get("per_query"), rb.get("per_query")
    if not pa or not pb:
        raise SystemExit("per_query 필드가 없습니다 — 결과를 재생성하세요")
    common = sorted(set(pa) & set(pb))
    if len(common) != len(pa) or len(common) != len(pb):
        print(f"경고: 공통 쿼리 {len(common)}건만 비교", file=sys.stderr)
    a = [pa[q][metric] for q in common]
    b = [pb[q][metric] for q in common]
    r = paired_bootstrap(a, b)
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    print(
        f"{ra['experiment']} - {rb['experiment']} ({metric}, n={r['n']}):\n"
        f"  mean diff = {r['mean_diff']:+.4f}, "
        f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}], "
        f"p = {r['p_value']:.4f}\n"
        f"  win/loss/tie = {wins}/{losses}/{r['n'] - wins - losses}"
    )


if __name__ == "__main__":
    main()
