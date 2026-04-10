from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import StrategyConfig, load_project_env
from .workflow import TechStrategyWorkflow


def _contains_expected_text(haystack: str, expected: str | list[str]) -> bool:
    """기대 문자열이 하나라도 검색 텍스트 안에 포함되는지 확인한다."""
    if isinstance(expected, list):
        return any(_normalize_text(item) in haystack for item in expected)
    return _normalize_text(expected) in haystack


def _normalize_text(value: str) -> str:
    """PDF 추출 텍스트의 공백, 대소문자, dash 표현 차이를 줄여 비교한다."""
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def is_relevant(retrieved_doc: dict[str, Any], item: dict[str, Any]) -> bool:
    """검색 문서가 기대 본문 근거와 기대 source 조건을 함께 만족하는지 확인한다."""
    haystack = _normalize_text(f'{retrieved_doc.get("title", "")}\n{retrieved_doc.get("content", "")}')
    if not _contains_expected_text(haystack, item["answer_source_contains"]):
        return False

    return is_expected_source(retrieved_doc, item)


def is_expected_source(retrieved_doc: dict[str, Any], item: dict[str, Any]) -> bool:
    """검색 문서가 기대 source 조건을 만족하는지 확인한다."""
    required_source = item.get("required_source_contains")
    if required_source:
        source_haystack = _normalize_text(f'{retrieved_doc.get("source", "")}\n{retrieved_doc.get("url", "")}')
        return _normalize_text(required_source) in source_haystack
    return True


def unique_by_source(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 PDF의 chunk가 Top-K를 채우지 않도록 source 기준으로 중복을 제거한다."""
    seen = set()
    deduped = []
    for doc in results:
        source = (doc.get("source") or doc.get("url") or doc.get("title") or "").lower()
        if source in seen:
            continue
        seen.add(source)
        deduped.append(doc)
    return deduped


def hit_rate_at_k(retrieved_by_query: list[list[dict[str, Any]]], eval_data: list[dict[str, Any]], k: int) -> float:
    """각 평가 질의의 상위 K개 chunk 안에 정답 본문 근거가 있는 비율을 계산한다."""
    hits = 0
    for item, results in zip(eval_data, retrieved_by_query, strict=False):
        if any(is_relevant(doc, item) for doc in results[:k]):
            hits += 1
    return hits / len(eval_data) if eval_data else 0.0


def source_hit_rate_at_k(
    retrieved_by_query: list[list[dict[str, Any]]],
    eval_data: list[dict[str, Any]],
    k: int,
) -> float:
    """각 평가 질의의 상위 K개 source 안에 기대 PDF가 있는 비율을 계산한다."""
    hits = 0
    for item, results in zip(eval_data, retrieved_by_query, strict=False):
        unique_results = unique_by_source(results)
        if any(is_expected_source(doc, item) for doc in unique_results[:k]):
            hits += 1
    return hits / len(eval_data) if eval_data else 0.0


def mrr_score(retrieved_by_query: list[list[dict[str, Any]]], eval_data: list[dict[str, Any]], k: int = 5) -> float:
    """정답 본문 근거가 처음 등장한 chunk 순위를 역순위 점수로 바꿔 평균 MRR을 계산한다."""
    reciprocal_ranks = []
    for item, results in zip(eval_data, retrieved_by_query, strict=False):
        rr = 0.0
        for rank, doc in enumerate(results[:k], start=1):
            if is_relevant(doc, item):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def source_mrr_score(retrieved_by_query: list[list[dict[str, Any]]], eval_data: list[dict[str, Any]], k: int = 5) -> float:
    """기대 PDF source가 처음 등장한 순위를 역순위 점수로 바꿔 평균 MRR을 계산한다."""
    reciprocal_ranks = []
    for item, results in zip(eval_data, retrieved_by_query, strict=False):
        rr = 0.0
        for rank, doc in enumerate(unique_by_source(results)[:k], start=1):
            if is_expected_source(doc, item):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def _first_rank(results: list[dict[str, Any]], item: dict[str, Any], *, mode: str, k: int) -> int | None:
    """정답 source 또는 본문 근거가 처음 등장한 순위를 찾는다."""
    candidates = unique_by_source(results) if mode == "source" else results
    matcher = is_expected_source if mode == "source" else is_relevant
    for rank, doc in enumerate(candidates[:k], start=1):
        if matcher(doc, item):
            return rank
    return None


def build_details(retrieved_by_query: list[list[dict[str, Any]]], eval_data: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """평가 질의별로 source rank와 evidence rank를 진단용으로 요약한다."""
    details = []
    for idx, (item, results) in enumerate(zip(eval_data, retrieved_by_query, strict=False), start=1):
        top_sources = []
        for doc in unique_by_source(results)[:k]:
            top_sources.append(
                {
                    "source": Path(str(doc.get("source", "local"))).name,
                    "score": doc.get("relevance_score"),
                    "title": doc.get("title", ""),
                }
            )
        details.append(
            {
                "id": idx,
                "question": item["question"],
                "expected_source": item.get("required_source_contains"),
                "expected_evidence": item["answer_source_contains"],
                "source_rank": _first_rank(results, item, mode="source", k=k),
                "evidence_rank": _first_rank(results, item, mode="evidence", k=k),
                "top_sources": top_sources,
            }
        )
    return details


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
    parser.add_argument(
        "--use-vector-store",
        action="store_true",
        help="Use cached FAISS vector store during evaluation. Defaults to disabled for corpus freshness.",
    )
    parser.add_argument("--k", type=int, default=5, help="Maximum cutoff K")
    parser.add_argument("--details", action="store_true", help="Include per-query source/evidence ranks in the JSON output.")
    return parser.parse_args()


def main() -> None:
    """평가 데이터를 로드해 적중률@K와 MRR을 JSON으로 출력하는 CLI 진입점."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    load_project_env(project_root)
    config = StrategyConfig.from_project_root(project_root)
    if not args.use_vector_store:
        config.enable_vector_store = False
    workflow = TechStrategyWorkflow(config)

    eval_file = Path(args.eval_file).resolve() if args.eval_file else project_root / "data" / "eval" / "retrieval_eval.sample.json"
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
    retrieved_by_query = [workflow._retrieve_documents(item["question"])[: args.k] for item in eval_data]

    result = {
        "retrieval": "hybrid_dense_lexical",
        "embedding": config.embedding_model,
        "eval_file": str(eval_file.relative_to(project_root)),
        "sample_size": len(eval_data),
        "metric_definition": "Hit Rate@K and MRR require the expected body evidence and source.",
        "vector_store_enabled": config.enable_vector_store,
        "Hit Rate@1": round(hit_rate_at_k(retrieved_by_query, eval_data, 1), 4),
        "Hit Rate@3": round(hit_rate_at_k(retrieved_by_query, eval_data, min(3, args.k)), 4),
        f"Hit Rate@{args.k}": round(hit_rate_at_k(retrieved_by_query, eval_data, args.k), 4),
        "MRR": round(mrr_score(retrieved_by_query, eval_data, args.k), 4),
        "diagnostic_source_hit": {
            "Hit Rate@1": round(source_hit_rate_at_k(retrieved_by_query, eval_data, 1), 4),
            "Hit Rate@3": round(source_hit_rate_at_k(retrieved_by_query, eval_data, min(3, args.k)), 4),
            f"Hit Rate@{args.k}": round(source_hit_rate_at_k(retrieved_by_query, eval_data, args.k), 4),
            "MRR": round(source_mrr_score(retrieved_by_query, eval_data, args.k), 4),
        },
    }
    if args.details:
        result["details"] = build_details(retrieved_by_query, eval_data, args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
