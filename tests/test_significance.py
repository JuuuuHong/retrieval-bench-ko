"""paired bootstrap 검정 검증."""

import pytest

from retrieval_bench.eval.significance import paired_bootstrap


class TestPairedBootstrap:
    def test_동일_시스템은_유의하지_않다(self):
        a = [0.5, 0.7, 0.3, 0.9] * 20
        r = paired_bootstrap(a, list(a))
        assert r["mean_diff"] == 0.0
        assert r["p_value"] == 1.0

    def test_압도적_차이는_유의하다(self):
        a = [1.0] * 50
        b = [0.0] * 50
        r = paired_bootstrap(a, b)
        assert r["mean_diff"] == 1.0
        assert r["p_value"] < 0.01
        assert r["ci95"][0] > 0

    def test_노이즈_수준_차이는_유의하지_않다(self):
        # 50쌍 중 1쌍만 다름 -> CI가 0을 포함해야 함
        a = [0.5] * 50
        b = [0.5] * 49 + [0.4]
        r = paired_bootstrap(a, b)
        assert r["ci95"][0] <= 0.0 <= r["ci95"][1] or r["p_value"] > 0.05

    def test_같은_시드는_같은_결과(self):
        a, b = [0.1, 0.9, 0.5] * 30, [0.2, 0.8, 0.5] * 30
        assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)

    def test_길이_불일치는_에러(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_빈_입력은_에러(self):
        with pytest.raises(ValueError):
            paired_bootstrap([], [])
