"""BM25용 토크나이저.

3주차 실험 변수: 공백 분리 vs 형태소 분석(Kiwi)이 BM25 재현율에 미치는 영향.
kiwipiepy는 무겁기 때문에 실제 사용 시점에 지연 임포트한다.
"""

from __future__ import annotations

from typing import Callable

Tokenizer = Callable[[str], list[str]]


def whitespace_tokenizer(text: str) -> list[str]:
    return text.lower().split()


def make_kiwi_tokenizer() -> Tokenizer:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()

    def tokenize(text: str) -> list[str]:
        return [token.form for token in kiwi.tokenize(text)]

    return tokenize


def get_tokenizer(name: str) -> Tokenizer:
    if name == "whitespace":
        return whitespace_tokenizer
    if name == "kiwi":
        return make_kiwi_tokenizer()
    raise ValueError(f"지원하지 않는 토크나이저: {name} (whitespace | kiwi)")
