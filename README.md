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

| experiment | recall@10 | recall@100 | ndcg@10 | tokens@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| bm25_baseline (공백 토큰화) | 0.4539 | 0.6732 | 0.3900 | 1,559 | 44.4 | 102.8 |
| bm25_kiwi (형태소 분석) | 0.5840 | 0.8935 | 0.4630 | 1,900 | 142.9 | 300.1 |
| dense_e5_small (mE5-small) | **0.8558** | 0.9826 | **0.7288** | 1,795 | 14.1 | 26.4 |
| hybrid_rrf (kiwi+dense, k=60) | 0.8282 | **0.9873** | 0.6848 | 1,981 | 179.0 | 464.7 |

*tokens@10 = 상위 10개 문서의 평균 LLM 입력 토큰(tiktoken o200k 프록시). 표는 `python -m retrieval_bench.report`로 생성.*

### 발견

1. **한국어 BM25에서 토큰화는 부차 요소가 아니다** — 형태소 분석(Kiwi)만으로 Recall@100
   **+22.2%p** (0.673 → 0.894). 교착어에서 조사·어미가 붙은 표층형을 그대로 매칭하는
   공백 토큰화는 어휘 불일치로 정답을 놓친다. 대가는 레이턴시 3.2배.
2. **하이브리드가 항상 이기지는 않는다** — RRF(k=60)는 Recall@100에서만 최고(0.9873)이고,
   **Recall@10에서는 dense 단독보다 낮다**(0.828 < 0.856). 약한 검색기(BM25)가 상위권
   순위를 끌어내리는 구조. "하이브리드 = 무조건 개선"이 아니라 k 예산과 rrf_k 튜닝의
   문제라는 것 — 3주차 스윕에서 정량화 예정.
3. **비용 축은 품질 축과 독립적으로 움직인다** — 형태소 BM25는 공백 대비 토큰을 22% 더
   쓰고(1,559 → 1,900), 하이브리드가 가장 비싸다(1,981). 재현율만 보고 검색기를 고르면
   LLM 입력 비용이 조용히 늘어난다.

> 주의: 코퍼스 서브셋(5만/148만) 기준이므로 절대값은 전체 코퍼스보다 낙관적이다.
> 검색기 간 상대 비교가 목적이며, 전체 코퍼스 측정은 인덱스 교체(bm25s/FAISS) 후 예정.

## 로드맵

- [x] 1주차: BM25 + MIRACL-ko + Recall@k 파이프라인 end-to-end
- [x] 2주차: 벡터 검색(mE5-small) + RRF 융합 + 토큰 비용 축, 4-way 비교표
- [ ] 3주차: 한국어 특화 실험 — 토큰화 방식 비교, RRF k 스윕, 청크 크기별 재현율·토큰비용 트레이드오프
- [ ] 4주차: cross-encoder 리랭킹, 결과 차트 자동 생성

## 테스트

메트릭 구현은 전부 수계산 값과 대조하는 유닛테스트를 거칩니다.
측정 도구가 틀리면 모든 결론이 무효이기 때문입니다.

```bash
pytest
```
