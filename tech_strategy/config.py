from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(search_root: Path | None = None) -> str:
    """가장 가까운 프로젝트 .env 파일을 로드하고 Matplotlib 캐시 설정을 준비한다."""
    search_root = (search_root or Path.cwd()).resolve()
    roots = [search_root, *search_root.parents]

    env_path = ""
    for root in roots:
        for candidate_name in (".env", ".env.example"):
            candidate = root / candidate_name
            if candidate.exists():
                env_path = str(candidate)
                break
        if env_path:
            break

    if not env_path:
        for root in roots:
            for candidate_name in (".env", ".env.example"):
                candidate = root / "langgraph-v1" / candidate_name
                if candidate.exists():
                    env_path = str(candidate)
                    break
            if env_path:
                break

    if env_path:
        load_dotenv(env_path, override=False)

    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
        os.environ.setdefault("LANGSMITH_TRACING", "false")

    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mini-project-mpl-")

    return env_path


@dataclass(slots=True)
class StrategyConfig:
    """워크플로우, 검색, 출력 단계에서 공유하는 설정값."""

    project_root: Path
    data_dir: Path
    output_dir: Path
    deliverable_label: str = field(
        default_factory=lambda: os.getenv(
            "TS_DELIVERABLE_LABEL",
            "3반_배석현+박나연",
        )
    )
    technology_catalog: tuple[str, ...] = ("HBM4", "HBM", "PIM", "CXL")
    competitor_catalog: tuple[str, ...] = (
        "SK hynix",
        "Samsung",
        "Micron",
        "NVIDIA",
        "AMD",
        "Intel",
    )
    planner_model: str = field(default_factory=lambda: os.getenv("TS_PLANNER_MODEL", "gpt-4.1-mini"))
    analysis_model: str = field(default_factory=lambda: os.getenv("TS_ANALYSIS_MODEL", "gpt-4.1"))
    draft_model: str = field(default_factory=lambda: os.getenv("TS_DRAFT_MODEL", "gpt-4.1"))
    embedding_model: str = field(default_factory=lambda: os.getenv("TS_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"))
    enable_dense_retrieval: bool = field(default_factory=lambda: os.getenv("TS_ENABLE_DENSE_RETRIEVAL", "0") in {"1", "true", "TRUE", "yes", "YES"})
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("TS_RETRIEVAL_TOP_K", "8")))
    retrieval_score_threshold: float = field(default_factory=lambda: float(os.getenv("TS_RETRIEVAL_SCORE_THRESHOLD", "0.8")))
    tavily_max_results: int = field(default_factory=lambda: int(os.getenv("TS_TAVILY_MAX_RESULTS", "5")))
    max_web_queries: int = field(default_factory=lambda: int(os.getenv("TS_MAX_WEB_QUERIES", "6")))
    min_retrieved_docs: int = field(default_factory=lambda: int(os.getenv("TS_MIN_RETRIEVED_DOCS", "4")))
    min_web_results: int = field(default_factory=lambda: int(os.getenv("TS_MIN_WEB_RESULTS", "6")))
    min_source_diversity: int = field(default_factory=lambda: int(os.getenv("TS_MIN_SOURCE_DIVERSITY", "2")))
    min_recent_ratio: float = field(default_factory=lambda: float(os.getenv("TS_MIN_RECENT_RATIO", "0.5")))
    min_source_reliability_score: float = field(default_factory=lambda: float(os.getenv("TS_MIN_SOURCE_RELIABILITY", "0.7")))
    max_bias_risk_score: float = field(default_factory=lambda: float(os.getenv("TS_MAX_BIAS_RISK", "0.65")))
    max_iteration: int = field(default_factory=lambda: int(os.getenv("TS_MAX_ITERATION", "5")))

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "StrategyConfig":
        """프로젝트 기본 데이터/출력 디렉터리를 기준으로 설정 객체를 만든다."""
        root = Path(project_root).resolve()
        data_dir = root / "data" / "knowledge_base"
        output_dir = root / "output"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(project_root=root, data_dir=data_dir, output_dir=output_dir)
