# retrieval-bench-ko

한국어 하이브리드 검색의 각 구성요소(토큰화, BM25, 벡터 검색, RRF 융합)가
**검색 품질과 비용에 얼마나 기여하는지** 통제된 실험으로 측정하는 평가 하네스.

## 왜 만들었나

RAG·검색 시스템을 만드는 팀은 많지만, "하이브리드 검색이 얼마나 좋아졌는가"를
재현 가능한 숫자로 답하는 팀은 드뭅니다. 특히 한국어는:

- 교착어 특성상 토큰화 방식(공백 vs 형태소)이 BM25 성능을 크게 좌우하는데, 정량 자료가 희소하고
- 영어 벤치마크(BEIR 등) 결과가 그대로 이식되지 않습니다

이 리포는 **모든 수치가 config(YAML) + 커밋 해시로 재현 가능**한 것을 원칙으로,
구성요소별 기여도를 하나씩 측정합니다.

## 설계

```
configs/     실험 1개 = YAML 1개 (데이터셋, retriever, 평가 k)
src/
  retrievers/  BM25 / (2주차) Dense / RRF — 공통 인터페이스
  eval/        Recall@k, MRR, nDCG (수계산 유닛테스트로 검증)
  data/        MIRACL-ko 로더 (positive+hard negative 풀 + 무작위 distractor)
  report/      결과 → 마크다운 표
results/     실행마다 config 전문 + 메트릭 JSON 저장
```

- **데이터셋**: MIRACL-ko — 한국어 위키 기반, 사람이 라벨한 qrels 포함.
  원본 `miracl/miracl`은 스크립트 기반이라 datasets 3.x+에서 로드 불가하여,
  BEIR 포맷 표준 Parquet 미러인 [taeminlee/Ko-miracl](https://huggingface.co/datasets/taeminlee/Ko-miracl)을 사용
- **메트릭**: Recall@k(주 지표), MRR@k, nDCG@k, 검색 레이턴시 p50/p95
- **코퍼스**: 전체 150만 건이 아닌 서브셋(정답+하드네거티브+무작위 5만 건)으로 측정하며, 모든 결과표에 코퍼스 크기를 명시

## 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m retrieval_bench.run configs/bm25_baseline.yaml
python -m retrieval_bench.run configs/bm25_kiwi.yaml
```

## 결과

> MIRACL-ko dev 213 쿼리, 코퍼스 서브셋 50,000건(정답+하드네거티브+무작위, seed=42).
> 모든 수치는 `configs/`의 YAML + 커밋 해시로 재현 가능. 원본 JSON은 실행 시 `results/`에 생성.

| experiment | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| bm25_baseline (공백 토큰화) | 0.4539 | 0.6720 | 0.4493 | 0.3894 | 45.9 | 106.0 |
| bm25_kiwi (형태소 분석) | **0.5840** | **0.8935** | 0.4630 | 0.4630 | 160.0 | 347.5 |

**첫 번째 발견**: 한국어 BM25에서 토큰화 방식은 부차적 요소가 아니다.
형태소 분석(Kiwi)만으로 Recall@100이 **+22.2%p** (0.672 → 0.894) — 교착어에서 조사·어미가
붙은 표층형을 그대로 매칭하는 공백 토큰화는 어휘 불일치(lexical mismatch)로 정답을 놓친다.
대가는 레이턴시 3.5배(p50 46ms → 160ms). 이 트레이드오프의 정량화가 이 리포의 목적이다.

## 로드맵

- [x] 1주차: BM25 + MIRACL-ko + Recall@k 파이프라인 end-to-end
- [ ] 2주차: 벡터 검색(BGE-M3) + RRF 융합, 3-way 비교표
- [ ] 3주차: 한국어 특화 실험 — 토큰화 방식 비교, RRF k 스윕, 청크 크기별 재현율·토큰비용 트레이드오프
- [ ] 4주차: cross-encoder 리랭킹, 결과 차트 자동 생성

## 테스트

메트릭 구현은 전부 수계산 값과 대조하는 유닛테스트를 거칩니다.
측정 도구가 틀리면 모든 결론이 무효이기 때문입니다.

```bash
pytest
```
