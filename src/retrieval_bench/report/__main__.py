"""results/*.json 전체를 README용 비교표로 출력.

사용법:
    python -m retrieval_bench.report            # results/ 전체, 파일명 순
    python -m retrieval_bench.report a.json b.json   # 지정 순서대로
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrieval_bench.report.markdown import combined_table


def main() -> None:
    paths = (
        [Path(p) for p in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(Path("results").glob("*.json"))
    )
    if not paths:
        print("results/에 결과 JSON이 없습니다", file=sys.stderr)
        raise SystemExit(1)
    results = [json.loads(p.read_text()) for p in paths]
    print(combined_table(results))


if __name__ == "__main__":
    main()
