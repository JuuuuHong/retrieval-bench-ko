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
- **코퍼스**: 전체 1,486,752건이 아닌 서브셋(정답+하드네거티브+무작위 5만 건)으로 측정하며, 모든 결과표에 코퍼스 크기를 명시

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

| experiment | corpus | recall@10 | recall@100 | ndcg@10 | tokens@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| bm25_baseline (공백) | 50,000 | 0.452 | 0.672 | 0.391 | 1,550 | 38.6 | 85.7 |
| bm25_whitespace_norm (공백+구두점 제거) | 50,000 | 0.452 | 0.682 | 0.392 | 1,525 | 38.7 | 112.8 |
| bm25_kiwi (형태소) | 50,000 | 0.581 | 0.894 | 0.458 | 1,986 | 151.1 | 374.1 |
| dense_e5_small | 50,000 | 0.863 | 0.984 | 0.734 | 1,785 | 17.5 | 29.1 |
| hybrid_rrf (무가중, k=60) | 50,000 | 0.834 | 0.983 | 0.686 | 2,002 | 174.6 | 429.3 |
| hybrid_rrf_weighted (dense×3) | 50,000 | **0.888**† | 0.988 | 0.734 | 1,904 | 171.3 | 400.1 |

*tokens@10 = 상위 10개 문서의 평균 LLM 입력 토큰(tiktoken o200k 프록시).
표는 `python -m retrieval_bench.report` 출력 기반이며, 근거 JSON은 `results/`에 커밋
(git SHA·라이브러리 버전·per-query 메트릭 포함). 볼드는 paired bootstrap(n=213,
10k resample)에서 차순위 대비 **유의한(p<0.05)** 우위만 표시한다.
†dense 대비 +2.5%p, 95% CI [+0.5, +4.6], p=0.013.*

### 발견

1. **한국어 BM25에서 토큰화는 부차 요소가 아니다** — 형태소 분석(Kiwi)은 구두점
   정규화를 통제한 뒤에도(`whitespace_norm` 대비) Recall@100 **+21.1%p**
   (95% CI [+15.6, +26.8], p<0.0001). 교착어에서 조사·어미가 붙은 표층형을 그대로
   매칭하는 공백 토큰화는 어휘 불일치로 정답을 놓친다. 구두점 제거 몫은 +1.0%p뿐.
   (대소문자 폴딩은 전 조건 동일 적용)
2. **무가중 RRF는 이 설정에서 이득이 없고, 그 손해는 상위 순위에서 발생한다** —
   dense와의 recall@10 차이(0.834 vs 0.863)는 유의하지 않고(p=0.080, 213쿼리 중
   154 동점), recall@100은 완전 동률(p=0.92). 유일하게 유의한 것은 **ndcg@10
   −0.048(p=0.005)** — 약한 lexical arm이 가중치 없이 융합될 때 상위 순위 품질을
   흐린다. 참고: 동점 처리 버그 수정 전에는 "hybrid가 @100 최고"로 보였다 —
   측정 버그 하나가 결론을 뒤집은 사례.
3. **하이브리드가 나쁜 게 아니라 무가중이 나빴다** — 발견 2의 진단(약한 검색기가
   상위권을 끌어내림)에 처방(dense 가중 ×3)을 적용하자 **recall@10에서 dense
   단독을 유의하게 역전**(+2.5%p, p=0.013, win/loss 26:9)했고 ndcg@10은 동률.
   즉 이 설정에서 하이브리드의 가치는 "융합 여부"가 아니라 **가중치 설계**에 있다.
   가중치 스윕과 rrf_k 스윕은 3주차.
4. ~~비용 축은 품질 축과 독립적으로 움직인다~~ — 초기 주장 철회. tokens@10은
   검색기의 특성이 아니라 "어떤 문서를 가져왔는가"의 부산물이며, 토큰당 재현율로
   보면 kiwi(0.29/1k tok)가 공백(0.29)과 대등하고 dense(0.48)가 가장 효율적이다.
   **고정 토큰 예산 하 재현율**(실제 RAG 의사결정 형태)로 재설계 예정.

### 한계 — 먼저 밝혀둠

이 표가 주장할 수 없는 것들:

1. **dense의 수치는 zero-shot이 아니다.** mE5-small은 2단계 지도학습에
   **MIRACL train(16개 언어, 40k쌍)을 포함**한다([모델 카드](https://huggingface.co/intfloat/multilingual-e5-small)).
   dev 쿼리 자체는 학습에 없지만 같은 코퍼스·같은 어노테이션 방식의 train으로
   파인튜닝된 in-domain 모델이다. BM25는 학습이 없으므로, 이 표는
   "이 벤치마크에서 dense가 이겼다"가 아니라 **"이 벤치마크 계열로 학습된 dense가
   이겼다"**로 읽어야 한다. MIRACL 미학습 한국어 임베딩 대조군을 추가 예정.
2. **서브셋(5만/148.7만) 절대값은 낙관적이며, "상대 순위가 보존된다"는 것도
   검증 대상인 가정이다.** distractor의 94%가 무작위 위키 문서라 대규모
   코퍼스에서 나타나는 어려운 오답(의미 유사·어휘 중복)이 과소 대표된다.
   코퍼스 크기 스윕으로 감쇠 곡선을 정량화할 예정.
3. **213 쿼리 규모에서 작은 차이는 노이즈다.** 시스템 간 비교는
   `python -m retrieval_bench.compare`(paired bootstrap, per-query 저장 기반)로
   검정하며, 유의하지 않은 차이는 발견으로 취급하지 않는다.
4. **레이턴시는 검색 방식이 아니라 구현 성숙도를 포함한다** — BM25는 순수 파이썬
   `rank_bm25`, dense는 numpy 전수 내적(BLAS). 절대 비교 금지, bm25s/FAISS 교체 후
   재측정 예정. dense의 인덱싱 비용은 임베딩 캐시 히트 시 기록되지 않는다
   (결과 JSON의 `index_cache_hit` 참조; 캐시 미스 시 5만 건 인코딩은 수 분 규모).

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
