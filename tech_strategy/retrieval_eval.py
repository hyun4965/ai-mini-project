from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import StrategyConfig, load_project_env
from .workflow import TechStrategyWorkflow


def is_relevant(retrieved_doc: dict[str, Any], answer_source_contains: str) -> bool:
    """검색 문서의 제목 또는 본문에 기대 근거 문자열이 포함되어 있는지 확인한다."""
    haystack = f'{retrieved_doc.get("title", "")}\n{retrieved_doc.get("content", "")}'.lower()
    return answer_source_contains.lower() in haystack


def hit_rate_at_k(workflow: TechStrategyWorkflow, eval_data: list[dict[str, Any]], k: int) -> float:
    """각 평가 질의의 상위 K개 검색 결과 안에 정답 근거가 있는 비율을 계산한다."""
    hits = 0
    for item in eval_data:
        results = workflow._retrieve_documents(item["question"])[:k]
        if any(is_relevant(doc, item["answer_source_contains"]) for doc in results):
            hits += 1
    return hits / len(eval_data) if eval_data else 0.0


def mrr_score(workflow: TechStrategyWorkflow, eval_data: list[dict[str, Any]], k: int = 5) -> float:
    """정답 근거가 처음 등장한 순위를 역순위 점수로 바꿔 평균 MRR을 계산한다."""
    reciprocal_ranks = []
    for item in eval_data:
        results = workflow._retrieve_documents(item["question"])[:k]
        rr = 0.0
        for rank, doc in enumerate(results, start=1):
            if is_relevant(doc, item["answer_source_contains"]):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def parse_args() -> argparse.Namespace:
    """검색 평가 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval with Hit Rate@K and MRR.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root for mini_project",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Path to retrieval evaluation JSON. Defaults to data/eval/retrieval_eval.sample.json",
    )
    parser.add_argument("--k", type=int, default=5, help="Maximum cutoff K")
    return parser.parse_args()


def main() -> None:
    """평가 데이터를 로드해 적중률@K와 MRR을 JSON으로 출력하는 CLI 진입점."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    load_project_env(project_root)
    config = StrategyConfig.from_project_root(project_root)
    workflow = TechStrategyWorkflow(config)

    eval_file = Path(args.eval_file).resolve() if args.eval_file else project_root / "data" / "eval" / "retrieval_eval.sample.json"
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))

    result = {
        "retrieval": "hybrid_dense_lexical",
        "embedding": config.embedding_model,
        "Hit@1": round(hit_rate_at_k(workflow, eval_data, 1), 4),
        "Hit@3": round(hit_rate_at_k(workflow, eval_data, min(3, args.k)), 4),
        f"Hit@{args.k}": round(hit_rate_at_k(workflow, eval_data, args.k), 4),
        "MRR": round(mrr_score(workflow, eval_data, args.k), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
