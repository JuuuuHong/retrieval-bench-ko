"""실험 러너.

사용법:
    python -m retrieval_bench.run configs/bm25_baseline.yaml

print는 flush=True — 백그라운드 실행(stdout 리다이렉트) 시 블록 버퍼링으로
진행 로그가 안 보이는 문제 방지.

원칙: 결과 JSON에는 config 전문과 소요 시간을 함께 기록해,
숫자만 보고도 어떤 설정에서 나왔는지 재현할 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from retrieval_bench.data.miracl import load_miracl_ko
from retrieval_bench.eval.cost import mean_topk_tokens
from retrieval_bench.eval.metrics import aggregate
from retrieval_bench.report.markdown import metrics_table
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.tokenizers import get_tokenizer


def _build_one(r: dict):
    if r["type"] == "bm25":
        return BM25Retriever(
            tokenizer=get_tokenizer(r.get("tokenizer", "whitespace")),
            k1=r.get("k1", 0.9),
            b=r.get("b", 0.4),
        )
    if r["type"] == "dense":
        from retrieval_bench.retrievers.dense import DenseRetriever

        return DenseRetriever(
            model_name=r["model"],
            query_prefix=r.get("query_prefix", ""),
            passage_prefix=r.get("passage_prefix", ""),
            batch_size=r.get("batch_size", 64),
        )
    if r["type"] == "rrf":
        from retrieval_bench.retrievers.rrf import RRFRetriever

        return RRFRetriever(
            components=[_build_one(c) for c in r["components"]],
            rrf_k=r.get("rrf_k", 60),
            candidate_k=r.get("candidate_k", 1000),
        )
    raise ValueError(f"지원하지 않는 retriever: {r['type']}")


def build_retriever(cfg: dict):
    return _build_one(cfg["retriever"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    experiment = cfg["experiment"]
    k_values = cfg["eval"]["k"]
    max_k = max(k_values)

    d = cfg["dataset"]
    print(f"[{experiment}] 데이터셋 로드 중: {d['name']} ({d['split']})", flush=True)
    dataset = load_miracl_ko(
        split=d["split"],
        corpus_sample=d.get("corpus_sample", 0),
        seed=d.get("seed", 42),
    )
    print(f"  쿼리 {len(dataset.queries)}건 / 코퍼스 {len(dataset.corpus)}건", flush=True)

    retriever = build_retriever(cfg)
    t0 = time.perf_counter()
    retriever.index(dataset.corpus)
    index_sec = time.perf_counter() - t0
    print(f"  인덱싱 완료: {index_sec:.1f}s", flush=True)

    runs: dict[str, list[str]] = {}
    latencies: list[float] = []
    for qid, query in tqdm(dataset.queries.items(), desc="검색"):
        t0 = time.perf_counter()
        results = retriever.search(query, k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        runs[qid] = [doc_id for doc_id, _ in results]

    metrics = aggregate(runs, dataset.qrels, k_values)
    metrics["latency_p50_ms"] = statistics.median(latencies)
    metrics["latency_p95_ms"] = statistics.quantiles(latencies, n=20)[18]

    cost_k = cfg["eval"].get("cost_k", 10)
    metrics[f"tokens@{cost_k}"] = mean_topk_tokens(runs, dataset.corpus, cost_k)

    out = {
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "corpus_size": len(dataset.corpus),
        "index_sec": index_sec,
        "metrics": metrics,
    }
    out_path = Path("results") / f"{experiment}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\n결과 저장: {out_path}\n", flush=True)
    print(metrics_table(experiment, metrics), flush=True)


if __name__ == "__main__":
    main()
