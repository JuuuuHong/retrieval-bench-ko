"""토크나이저 정책 검증 — 특히 비교 실험의 통제 변수(소문자·구두점)."""

from retrieval_bench.tokenizers import (
    whitespace_norm_tokenizer,
    whitespace_tokenizer,
)


class TestWhitespace:
    def test_소문자화(self):
        assert whitespace_tokenizer("IMF 외환위기") == ["imf", "외환위기"]

    def test_구두점은_유지(self):
        assert whitespace_tokenizer("무엇인가?") == ["무엇인가?"]


class TestWhitespaceNorm:
    def test_구두점_제거(self):
        assert whitespace_norm_tokenizer("무엇인가?") == ["무엇인가"]

    def test_구두점이_토큰을_분리(self):
        assert whitespace_norm_tokenizer("한국(대한민국)의 수도") == [
            "한국",
            "대한민국",
            "의",
            "수도",
        ]

    def test_소문자화_동일_적용(self):
        assert whitespace_norm_tokenizer("DNA 구조") == ["dna", "구조"]
