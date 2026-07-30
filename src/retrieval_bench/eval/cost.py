"""토큰 비용 측정.

검색 품질이 같아도 상위 k 문서가 길면 LLM 입력 비용이 커진다.
BM25는 길이 정규화(b 파라미터)로 짧은 문서를, 벡터 검색은 의미 밀도가 높은
문서를 선호하는 경향이 있어 검색기별 토큰 비용이 다를 수 있다 — 그 차이를 잰다.

토크나이저는 tiktoken o200k_base(GPT-4o 계열)를 프록시로 사용한다.
Claude 등 다른 모델과 절대값은 다르지만, 검색기 간 상대 비교에는 충분하다.
"""

from __future__ import annotations

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("o200k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def mean_topk_tokens(
    runs: dict[str, list[str]],
    corpus: dict[str, str],
    k: int,
) -> float:
    """쿼리별 상위 k 문서 토큰 합의 평균 — 'LLM에 넣을 때 쿼리당 평균 비용'."""
    if not runs:
        raise ValueError("runs가 비어 있습니다")
    # 코퍼스에 없는 문서는 조용히 0토큰 처리하지 않는다 — 망가진 검색기가
    # "저렴해 보이는" 왜곡을 만들기 때문. KeyError를 그대로 전파한다.
    totals = [
        sum(count_tokens(corpus[d]) for d in ranked[:k])
        for ranked in runs.values()
    ]
    return sum(totals) / len(totals)
