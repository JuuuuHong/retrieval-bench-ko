"""생성(RAG) 단계 평가 — 검색 품질 차이가 최종 답변의 근거성으로 이어지는가.

검색 메트릭(recall@k)은 "정답 문서가 상위에 왔는가"만 본다. 실제 RAG에서
중요한 것은 그 다음이다:
1. 검색이 정답 문서를 가져왔을 때, 생성 모델이 그 문서를 근거로 인용해 답하는가
2. 검색이 정답 문서를 못 가져왔을 때, 지어내지 않고 "근거 없음"을 내는가

정답 답변 텍스트가 없는 데이터셋(MIRACL은 문서 관련성 라벨만 있다)에서도
사람 채점이나 모델 채점 없이 재현 가능하게 재기 위해, 평가를 **인용**과
**기권**으로만 정의한다. 문서 관련성은 사람이 라벨한 qrels가 판정한다.

조건:
- retrieved: 검색기 상위 n_ctx 문서를 그대로 준다(실제 파이프라인)
- negatives_only: 사람이 "무관"으로 판정한 문서(MIRACL qrels score=0)만 준다.
  주제는 비슷하지만 답이 없는 문서라, 여기서의 답변은 근거 없는 답변이다.
  (무작위 distractor는 판정되지 않아 답을 담고 있을 수 있으므로 쓰지 않는다)

메트릭(쿼리별 0/1, 평균):
- answered: "근거 없음"이 아닌 답을 냈다
- grounded: 답했고, 인용한 문서 중 하나 이상이 정답 문서다
- cited_gold_precision: 인용한 문서 중 정답 문서 비율(인용이 있는 쿼리만)
- unsupported: 답했지만 인용 문서 중 정답이 하나도 없다
- context_has_gold: 컨텍스트에 정답 문서가 하나라도 있었다(검색 단계의 몫)

생성 모델은 로컬 transformers 모델만 쓴다(외부 API 호출 없음). greedy 디코딩이라
같은 모델·같은 입력이면 같은 출력이 나오고, 출력 원문을 결과에 저장한다.

사용법:
    python -m retrieval_bench.generate configs/gen/miracl_50k.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ABSTAIN = "근거 없음"

SYSTEM_PROMPT = (
    "너는 주어진 문서만 근거로 한국어 질문에 답하는 도우미다.\n"
    "규칙:\n"
    "1. 문서에 있는 내용만 사용한다. 너의 사전 지식은 쓰지 않는다.\n"
    "2. 답의 근거가 된 문서 번호를 [1], [2]처럼 문장 끝에 표시한다.\n"
    f"3. 문서에 답이 없으면 다른 말 없이 \"{ABSTAIN}\"이라고만 쓴다.\n"
    "4. 두세 문장 이내로 답한다."
)

_CITE = re.compile(r"\[(\d+)\]")


def build_user_prompt(question: str, docs: list[str], max_chars: int) -> str:
    blocks = [f"[{i}] {text[:max_chars]}" for i, text in enumerate(docs, start=1)]
    return "문서:\n" + "\n\n".join(blocks) + f"\n\n질문: {question}"


def parse_answer(text: str, n_docs: int) -> dict:
    """모델 출력 → 기권 여부와 인용 번호(1-base, 범위 밖 번호는 invalid로 분리)."""
    stripped = text.strip()
    cited = [int(m) for m in _CITE.findall(stripped)]
    valid = sorted({c for c in cited if 1 <= c <= n_docs})
    invalid = sorted({c for c in cited if not 1 <= c <= n_docs})
    # 기권 문구로 시작하고 인용이 없을 때만 기권으로 본다 — "근거 없음"이라 해놓고
    # 뒤에 답을 덧붙이는 출력을 기권으로 세면 기권률이 부풀려진다
    abstained = stripped.startswith(ABSTAIN) and not valid
    return {"abstained": abstained, "cited": valid, "invalid_cited": invalid}


def score_one(parsed: dict, ctx_ids: list[str], gold: set[str]) -> dict:
    cited_ids = [ctx_ids[c - 1] for c in parsed["cited"]]
    cited_gold = [d for d in cited_ids if d in gold]
    answered = not parsed["abstained"]
    out = {
        "answered": float(answered),
        "grounded": float(answered and bool(cited_gold)),
        "unsupported": float(answered and not cited_gold),
        "context_has_gold": float(any(d in gold for d in ctx_ids)),
        "invalid_citation": float(bool(parsed["invalid_cited"])),
    }
    if cited_ids:
        out["cited_gold_precision"] = len(cited_gold) / len(cited_ids)
    return out


def aggregate_scores(per_query: dict[str, dict]) -> dict:
    keys = ["answered", "grounded", "unsupported", "context_has_gold", "invalid_citation"]
    n = len(per_query)
    agg = {k: sum(v[k] for v in per_query.values()) / n for k in keys}
    with_cite = [v["cited_gold_precision"] for v in per_query.values() if "cited_gold_precision" in v]
    agg["cited_gold_precision"] = sum(with_cite) / len(with_cite) if with_cite else None
    agg["num_queries"] = n
    # 검색이 정답을 가져온 쿼리만 놓고 본 생성 단계의 몫
    hit = [v for v in per_query.values() if v["context_has_gold"]]
    agg["grounded_given_gold_in_context"] = (
        sum(v["grounded"] for v in hit) / len(hit) if hit else None
    )
    miss = [v for v in per_query.values() if not v["context_has_gold"]]
    agg["abstain_given_no_gold"] = (
        sum(1 - v["answered"] for v in miss) / len(miss) if miss else None
    )
    return agg


class LocalGenerator:
    """transformers 로컬 생성기 — greedy, 채팅 템플릿 사용."""

    def __init__(self, model_name: str, max_new_tokens: int = 160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if device == "mps" else torch.float32
        ).to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def load_hard_negatives(split: str) -> dict[str, list[str]]:
    """MIRACL-ko qrels에서 사람이 무관(score=0)으로 판정한 문서."""
    from datasets import load_dataset

    out: dict[str, list[str]] = {}
    for row in load_dataset("taeminlee/Ko-miracl", "default", split=split):
        if float(row["score"]) == 0:
            out.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
    return {qid: sorted(docs) for qid, docs in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="디버그용 쿼리 수 제한")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from retrieval_bench.run import _git_info, _versions, load_dataset_from_cfg

    dataset = load_dataset_from_cfg(cfg["dataset"])
    n_ctx = cfg.get("n_ctx", 5)
    max_chars = cfg.get("max_chars", 600)
    generator = LocalGenerator(cfg["generator"], cfg.get("max_new_tokens", 160))

    conditions: dict[str, dict[str, list[str]]] = {}
    for name, path in cfg.get("retrieved", {}).items():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            runs = json.load(f)["runs"]
        conditions[name] = {qid: docs[:n_ctx] for qid, docs in runs.items()}
    if cfg.get("negatives_only"):
        negs = load_hard_negatives(cfg["dataset"]["split"])
        conditions["negatives_only"] = {
            qid: [d for d in docs if d in dataset.corpus][:n_ctx]
            for qid, docs in negs.items()
        }

    qids = sorted(dataset.qrels)
    if args.limit:
        qids = qids[: args.limit]

    results = {}
    for name, ctx in conditions.items():
        per_query, outputs = {}, {}
        t0 = time.perf_counter()
        for qid in qids:
            ctx_ids = ctx.get(qid, [])
            if not ctx_ids:
                continue
            user = build_user_prompt(
                dataset.queries[qid], [dataset.corpus[d] for d in ctx_ids], max_chars
            )
            text = generator(SYSTEM_PROMPT, user)
            parsed = parse_answer(text, len(ctx_ids))
            per_query[qid] = score_one(parsed, ctx_ids, dataset.qrels[qid])
            outputs[qid] = {"ctx": ctx_ids, "output": text, **parsed}
        sec = time.perf_counter() - t0
        agg = aggregate_scores(per_query)
        agg["sec_per_query"] = sec / max(1, len(per_query))
        results[name] = {"metrics": agg, "per_query": per_query, "outputs": outputs}
        print(f"[{name}] " + " · ".join(
            f"{k}={v:.3f}" for k, v in agg.items() if isinstance(v, float)
        ), flush=True)

    ts = datetime.now(timezone.utc)
    out = {
        "experiment": cfg["experiment"],
        "timestamp": ts.isoformat(),
        "config": cfg,
        "system_prompt": SYSTEM_PROMPT,
        "corpus_size": len(dataset.corpus),
        "conditions": results,
        **_git_info(),
        "versions": {**_versions(), "transformers": __import__("transformers").__version__},
    }
    suffix = f"_limit{args.limit}" if args.limit else ""
    latest = Path("results/gen") / f"{cfg['experiment']}{suffix}.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"결과 저장: {latest}")


if __name__ == "__main__":
    main()
