from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryInterpretation(BaseModel):
    """분석 범위 추출과 검색 질의 생성을 위한 구조화된 planner 출력."""
    primary_technology: str = Field(description="분석의 중심이 되는 대표 기술")
    target_technologies: list[str] = Field(description="사용자 시나리오에서 추출한 분석 대상 기술 범위")
    target_competitors: list[str] = Field(description="비교 분석에 사용할 경쟁사 목록")
    reasoning: str = Field(description="해당 기술과 경쟁사를 선택한 이유")
    retrieval_queries: list[str] = Field(description="로컬 문서 검색에 사용할 질의 목록")
    web_queries: list[str] = Field(description="최신 외부 정보를 찾기 위한 웹 검색 질의 목록")
    counter_queries: list[str] = Field(description="반증 근거를 찾기 위한 검색 질의 목록")


class RetrievedDoc(BaseModel):
    """로컬 검색 문서를 워크플로우에서 쓰기 좋게 정규화한 레코드."""
    title: str
    source: str
    source_type: str = "retrieval"
    content: str
    relevance_score: float
    url: str | None = None
    metadata: dict = Field(default_factory=dict)


class NewsItem(BaseModel):
    """웹 검색 결과를 워크플로우에서 쓰기 좋게 정규화한 레코드."""
    title: str
    source: str
    url: str | None = None
    content: str
    query: str
    published_at: str | None = None
    stance: Literal["supportive", "counter", "neutral"] = "neutral"
    source_type: str = "web_search"


class AssessmentResult(BaseModel):
    """기술과 경쟁사 한 쌍에 대한 구조화된 평가 결과."""
    technology: str
    competitor: str
    direct_evidence: list[str]
    indirect_evidence: list[str]
    direct_evidence_count: int
    indirect_evidence_count: int
    evidence_quality_score: float
    evidence_completeness: bool
    evidence_summary: str
    current_status_summary: str
    competitor_level_summary: str
    trl_level: int
    trl_confidence: float
    trl_rationale: str
    commercialization_signal: Literal["Research", "Prototype", "Pilot", "Production", "Unclear"]
    relative_position_to_sk_hynix: Literal["Ahead", "Comparable", "Behind", "Unclear"]
    threat_level: Literal["High", "Medium", "Low"]
    threat_score: float
    market_impact: float
    competition_intensity: float
    threat_rationale: str
    strategic_implication: str
    uncertainty_note: str = ""


class DecisionRecommendation(BaseModel):
    """개별 기술에 대한 R&D 의사결정 추천."""
    technology: str
    rd_feasibility: Literal["Go", "Hold", "Monitor"]
    priority_level: Literal["High", "Medium", "Low"]
    decision_score: float
    decision_rationale: str
    is_action_required: bool
    suggested_actions: list[str] = Field(default_factory=list)
    target_competitors: list[str] = Field(default_factory=list)


class DecisionOutput(BaseModel):
    """Decision Agent가 반환하는 포트폴리오 관점의 의사결정 결과."""
    summary: str
    recommendations: list[DecisionRecommendation]
    portfolio_view: str
