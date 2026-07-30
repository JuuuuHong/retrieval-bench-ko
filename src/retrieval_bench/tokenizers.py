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
        # 소문자 폴딩을 whitespace_tokenizer와 동일하게 적용 —
        # 두 토크나이저 비교가 "형태소 분석 유무" 단일 변수가 되도록
        # (한국어 위키에는 IMF/DNA 등 로마자 표기가 흔해 대소문자 정책이
        #  통제되지 않으면 비교가 교란된다)
        return [token.form.lower() for token in kiwi.tokenize(text)]

    return tokenize


def get_tokenizer(name: str) -> Tokenizer:
    if name == "whitespace":
        return whitespace_tokenizer
    if name == "kiwi":
        return make_kiwi_tokenizer()
    raise ValueError(f"지원하지 않는 토크나이저: {name} (whitespace | kiwi)")
