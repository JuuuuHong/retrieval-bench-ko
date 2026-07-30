"""비교표 생성 검증."""

import pytest

from retrieval_bench.report.markdown import combined_table


def _result(name: str, **metrics):
    return {"experiment": name, "corpus_size": 50000, "metrics": metrics}


class TestCombinedTable:
    def test_두_실험_비교표(self):
        table = combined_table(
            [
                _result("a", **{"recall@10": 0.5, "tokens@10": 1500.0}),
                _result("b", **{"recall@10": 0.8, "tokens@10": 1600.0}),
            ],
            columns=["recall@10", "tokens@10"],
        )
        lines = table.splitlines()
        assert len(lines) == 4  # 헤더 + 구분선 + 행 2
        assert "| a | 50,000 | 0.5000 | 1500.0000 |" in table
        assert lines[0] == "| experiment | corpus | recall@10 | tokens@10 |"

    def test_없는_메트릭은_대시(self):
        table = combined_table(
            [_result("a", **{"recall@10": 0.5})],
            columns=["recall@10", "tokens@10"],
        )
        assert "| 0.5000 | - |" in table

    def test_빈_입력은_에러(self):
        with pytest.raises(ValueError):
            combined_table([])
