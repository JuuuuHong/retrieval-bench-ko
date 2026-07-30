"""paired bootstrap 유의성 검정.

213 쿼리 규모에서 시스템 간 평균 차이는 노이즈일 수 있다.
같은 쿼리 집합에서 측정한 per-query 점수 쌍을 리샘플링해
평균 차이의 95% CI와 양측 p값을 낸다.

표준 라이브러리만 사용(외부 통계 패키지 의존 없음) — CI에서도 돈다.
"""

from __future__ import annotations

import random


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """시스템 a와 b의 per-query 점수(같은 쿼리 순서)로 a-b 차이를 검정.

    반환: mean_diff(관측 평균 차), ci95(부트스트랩 95% CI),
          p_value(양측: 부트스트랩 분포가 0을 넘는 비율 기반), n(쿼리 수)
    """
    if len(a) != len(b):
        raise ValueError(f"쿼리 수 불일치: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("빈 입력")

    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    boots = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boots.append(total / n)
    boots.sort()

    ci_low = boots[int(0.025 * n_resamples)]
    ci_high = boots[int(0.975 * n_resamples) - 1]
    le_zero = sum(1 for x in boots if x <= 0)
    ge_zero = sum(1 for x in boots if x >= 0)
    p_value = min(1.0, 2 * min(le_zero, ge_zero) / n_resamples)

    return {
        "mean_diff": observed,
        "ci95": (ci_low, ci_high),
        "p_value": p_value,
        "n": n,
    }
