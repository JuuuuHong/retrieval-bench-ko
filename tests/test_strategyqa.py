"""Ko-StrategyQA qrels 파싱 검증 (다운로드 없이)."""

import io

from retrieval_bench.data.strategyqa import parse_qrels_tsv


def test_score_0은_정답이_아니다():
    tsv = io.StringIO(
        "query-id\tcorpus-id\tscore\n"
        "q1\tA-1\t1\n"
        "q1\tA-2\t1\n"
        "q2\tB-1\t0\n"
        "q3\tC-1\t1\n"
    )
    assert parse_qrels_tsv(tsv) == {"q1": {"A-1", "A-2"}, "q3": {"C-1"}}
