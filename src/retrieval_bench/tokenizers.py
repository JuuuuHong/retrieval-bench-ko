"""BM25용 토크나이저.

3주차 실험 변수: 공백 분리 vs 형태소 분석(Kiwi)이 BM25 재현율에 미치는 영향.
kiwipiepy는 무겁기 때문에 실제 사용 시점에 지연 임포트한다.
"""

from __future__ import annotations

import re
from typing import Callable

Tokenizer = Callable[[str], list[str]]


_PUNCT = re.compile(r"[^\w\s]")


def whitespace_tokenizer(text: str) -> list[str]:
    return text.lower().split()


def whitespace_norm_tokenizer(text: str) -> list[str]:
    """공백 분리 + 구두점 제거.

    형태소 분석 효과 주장에서 "구두점 정규화 몫"을 분리하기 위한 통제 조건.
    MIRACL-ko 쿼리는 전부 물음표 등 구두점 붙은 토큰을 포함하므로,
    이 조건 없이는 kiwi와의 비교가 straw man 의심을 받는다.
    """
    return _PUNCT.sub(" ", text.lower()).split()


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
    if name == "whitespace_norm":
        return whitespace_norm_tokenizer
    if name == "kiwi":
        return make_kiwi_tokenizer()
    raise ValueError(
        f"지원하지 않는 토크나이저: {name} (whitespace | whitespace_norm | kiwi)"
    )
