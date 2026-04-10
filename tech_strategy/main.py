from __future__ import annotations

import argparse
from pathlib import Path

from .config import StrategyConfig, load_project_env
from .state import create_initial_state
from .workflow import TechStrategyWorkflow


def _print_failure_diagnostics(final_state: dict) -> None:
    """실패 시 각 단계의 실패 이유와 마지막 실행 로그를 사람이 읽기 쉽게 출력한다."""
    if final_state["control"].get("status") == "success":
        return

    retrieval = final_state.get("retrieval", {})
    web_search = final_state.get("web_search", {})
    assessment = final_state.get("assessment", {})
    decision = final_state.get("decision", {})
    draft = final_state.get("draft", {})

    print("\nFailure diagnostics:")
    print(
        "- Retrieval: "
        f"success={retrieval.get('is_success')}, "
        f"reason={retrieval.get('failure_reason') or 'n/a'}, "
        f"docs={len(retrieval.get('filtered_docs', []))}, "
        f"confidence={retrieval.get('confidence', 0.0):.2f}"
    )
    print(
        "- Web Search: "
        f"success={web_search.get('is_success')}, "
        f"reason={web_search.get('failure_reason') or 'n/a'}, "
        f"results={len(web_search.get('web_results', []))}, "
        f"sources={web_search.get('source_diversity', 0)}, "
        f"recent={web_search.get('freshness_score', 0.0):.2f}, "
        f"reliability={web_search.get('source_reliability_score', 0.0):.2f}"
    )
    print(
        "- Downstream: "
        f"assessment={assessment.get('status') or 'n/a'} "
        f"({assessment.get('failure_reason') or 'n/a'}), "
        f"decision={decision.get('status') or 'n/a'} "
        f"({decision.get('failure_reason') or 'n/a'}), "
        f"draft={draft.get('status') or 'n/a'} "
        f"({draft.get('failure_reason') or 'n/a'})"
    )

    recent_logs = final_state.get("analysis_log", [])[-5:]
    if recent_logs:
        print("\nRecent logs:")
        for log in recent_logs:
            print(f"- {log}")


def parse_args() -> argparse.Namespace:
    """전체 기술 전략 워크플로우 실행에 필요한 CLI 옵션을 파싱한다."""
    parser = argparse.ArgumentParser(description="Run the technology strategy analysis workflow.")
    parser.add_argument("user_query", help="Technology scenario to analyze")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root for mini_project",
    )
    parser.add_argument("--data-dir", default=None, help="Optional override for retrieval corpus directory")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    parser.add_argument("--max-iteration", type=int, default=None, help="Maximum supervisor retry count")
    parser.add_argument(
        "--team-label",
        default=None,
        help="Deliverable label, for example 3반_배석현_박나연",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 인자대로 워크플로우를 실행하고 핵심 최종 상태값을 출력한다."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    env_path = load_project_env(project_root)
    config = StrategyConfig.from_project_root(project_root)

    if args.data_dir:
        config.data_dir = Path(args.data_dir).resolve()
        config.data_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        config.output_dir = Path(args.output_dir).resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_iteration is not None:
        config.max_iteration = args.max_iteration
    if args.team_label:
        config.deliverable_label = args.team_label

    workflow = TechStrategyWorkflow(config).build()
    initial_state = create_initial_state(args.user_query, config.max_iteration)
    final_state = workflow.invoke(initial_state)

    print(f"Loaded env: {env_path or 'not found'}")
    print(f"Status: {final_state['control']['status']}")
    print(f"Next step: {final_state['control']['next_step']}")
    print(f"Coverage: {final_state['control']['coverage_status']}")
    print(f"PDF: {final_state['output'].get('final_pdf_path', '') or '(not generated)'}")
    print(f"Deliverable label: {config.deliverable_label}")
    _print_failure_diagnostics(final_state)
    print("\nDecision summary:")
    print(final_state["decision"].get("result", {}).get("summary", "(no decision summary)"))


if __name__ == "__main__":
    main()
