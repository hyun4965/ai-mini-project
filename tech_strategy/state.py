from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """LangGraph가 노드의 부분 업데이트를 적용할 때 중첩 상태 그룹을 병합한다."""
    return {**(left or {}), **(right or {})}


class ScopeState(TypedDict):
    """워크플로우 노드들이 공유하는 분석 기술과 경쟁사 범위."""
    # 비교 기준이 되는 자사. 예: SK hynix
    subject_company: str
    # 대표 분석 기술. 예: HBM4
    target_technology: str
    # 분석 대상 기술 전체 목록. 예: HBM4, PIM, CXL
    target_technologies: list[str]
    # 기준 기업(subject_company)과 비교할 외부 경쟁사 목록. 예: Samsung, Micron
    target_competitors: list[str]


class RetrievalState(TypedDict):
    """로컬 문서 검색 결과와 재시도 관련 메타데이터."""
    # 현재 로컬 검색 노드가 기준으로 삼는 검색 질의.
    query: str
    # 검색 단계에서 일단 후보로 수집된 문서 목록.
    candidate_docs: list[dict[str, Any]]
    # 관련도 점수와 분석 범위 조건을 통과한 문서 목록.
    filtered_docs: list[dict[str, Any]]
    # 이후 평가 단계에서 실제 근거로 사용할 최종 검색 문서 목록.
    retrieved_docs: list[dict[str, Any]]
    # 검색 후보 문서들의 관련도 점수 목록.
    relevance_scores: list[float]
    # 최종 검색 문서들의 평균 관련도 기반 신뢰도.
    confidence: float
    # 로컬 검색이 최소 문서 수와 품질 기준을 통과했는지 여부.
    is_success: bool
    # 검색 실패 시 질의 재작성 방향을 정하기 위한 실패 사유.
    failure_reason: str
    # 로컬 검색 노드가 질의 재작성 후 재시도한 횟수.
    attempt: int


class WebSearchState(TypedDict):
    """외부 웹 검색 결과, 품질 지표, 재시도 관련 메타데이터."""
    # 실제 실행한 웹 검색 질의 목록.
    queries: list[str]
    # 평가 단계에서 사용할 최종 웹 검색 결과 목록.
    web_results: list[dict[str, Any]]
    # 검색 결과가 나온 서로 다른 출처 또는 도메인의 개수.
    source_diversity: int
    # 최신 자료 비율을 나타내는 점수.
    freshness_score: float
    # 출처 신뢰도 점수의 평균값.
    source_reliability_score: float
    # 긍정 근거뿐 아니라 반증/리스크 근거가 포함되어 있는지 여부.
    has_counter_evidence: bool
    # 특정 출처나 특정 기업에 결과가 과도하게 쏠린 정도.
    bias_risk_score: float
    # 경쟁사별 검색 결과 언급 횟수.
    competitor_coverage: dict[str, int]
    # 경쟁사 커버리지가 최소 기준 이상으로 균형 있게 확보됐는지 여부.
    balanced_company_coverage: bool
    # 웹 검색이 출처 다양성, 최신성, 신뢰도, 편향 기준을 통과했는지 여부.
    is_success: bool
    # 웹 검색 실패 시 질의 재작성 방향을 정하기 위한 실패 사유.
    failure_reason: str
    # 웹 검색 노드가 질의 재작성 후 재시도한 횟수.
    attempt: int


class AssessmentState(TypedDict):
    """근거 종합, TRL 평가, 위협도 평가 결과를 담는 상태."""
    # 기술/경쟁사별 직접 근거, 간접 지표, 출처를 묶은 근거 묶음.
    evidence_bundle: dict[str, Any]
    # 기술-경쟁사 쌍별 최종 평가 결과 목록.
    results: list[dict[str, Any]]
    # 모든 예상 기술-경쟁사 쌍에 대한 평가가 완료됐는지 여부.
    is_complete: bool
    # 평가 실패 또는 품질 검증 실패 사유.
    failure_reason: str


class DecisionState(TypedDict):
    """최종 R&D 추천 결과와 검증 상태."""
    # Go/Hold/Monitor, 우선순위, 판단 근거, 실행안을 포함한 최종 의사결정 결과.
    result: dict[str, Any]
    # 의사결정 결과가 형식, 근거 연결성, 실행안 기준을 통과했는지 여부.
    is_valid: bool
    # 의사결정 검증 실패 사유.
    failure_reason: str


class DraftState(TypedDict):
    """Markdown 보고서 초안, 품질 점수, 수정 필요 여부."""
    # 보고서 초안이자 PDF 변환에 사용할 Markdown 본문.
    markdown_text: str
    # 초안 생성 또는 재작성 버전 번호.
    version: int
    # 필수 섹션, 근거 연결성, 분석형 문체 등을 기준으로 계산한 품질 점수.
    quality_score: float
    # 관리 노드가 초안 재작성이 필요하다고 판단했는지 여부.
    needs_revision: bool
    # 초안이 보고서 품질 검증을 통과했는지 여부.
    is_valid: bool
    # 초안 검증 실패 사유.
    failure_reason: str


class OutputState(TypedDict):
    """생성된 산출물 경로와 포맷팅 오류 상태."""
    # 생성된 PDF 파일 경로.
    pdf_path: str
    # 최종 제출용 PDF 파일 경로.
    final_pdf_path: str
    # PDF 생성과 검증이 성공했는지 여부.
    is_pdf_generated: bool
    # PDF 변환 또는 검증 중 발생한 오류 메시지.
    format_error: str | None


class ControlState(TypedDict):
    """Supervisor 라우팅, 재시도 횟수, 워크플로우 진행 상태."""
    # 전체 워크플로우 상태. 예: running, completed, failed
    status: str
    # 현재 또는 직전에 실행된 워크플로우 단계.
    workflow_stage: str
    # retrieval/web search 정보 충분성 상태를 사람이 읽기 좋게 요약한 문자열.
    coverage_status: str
    # 검색과 웹 근거가 다음 분석 단계로 넘어갈 만큼 충분한지 여부.
    is_information_sufficient: bool
    # 관리 노드 기준 전체 재시도 횟수.
    retry_count: int
    # 관리 노드가 허용하는 최대 재시도 횟수.
    max_iteration: int
    # 관리 노드가 다음에 실행하도록 선택한 노드 이름.
    next_step: str
    # 종료 판단 값. 완료 또는 실패로 끝날 때 END로 설정된다.
    final_decision: str
    # 로컬 검색/웹 검색에서 질의를 재작성한 이력.
    query_rewrite_history: list[str]


class StrategyState(TypedDict):
    """도메인별 중첩 그룹으로 구성한 최상위 LangGraph 상태."""
    # 사용자가 처음 입력한 기술 전략 분석 요청.
    user_query: str
    # 분석 기술과 경쟁사 범위. 부분 업데이트 시 기존 값과 병합된다.
    scope: Annotated[ScopeState, merge_dict]
    # 계획 노드가 만든 로컬 검색/웹 검색/반증 검색 계획.
    query_plan: dict[str, Any]
    # 로컬 문서 검색 상태. 부분 업데이트 시 기존 값과 병합된다.
    retrieval: Annotated[RetrievalState, merge_dict]
    # 외부 웹 검색 상태. 부분 업데이트 시 기존 값과 병합된다.
    web_search: Annotated[WebSearchState, merge_dict]
    # 근거 종합, TRL, 위협도 평가 상태. 부분 업데이트 시 기존 값과 병합된다.
    assessment: Annotated[AssessmentState, merge_dict]
    # R&D 의사결정 상태. 부분 업데이트 시 기존 값과 병합된다.
    decision: Annotated[DecisionState, merge_dict]
    # 보고서 초안 작성 상태. 부분 업데이트 시 기존 값과 병합된다.
    draft: Annotated[DraftState, merge_dict]
    # 최종 산출물 생성 상태. 부분 업데이트 시 기존 값과 병합된다.
    output: Annotated[OutputState, merge_dict]
    # 관리 노드 라우팅과 재시도 제어 상태. 부분 업데이트 시 기존 값과 병합된다.
    control: Annotated[ControlState, merge_dict]
    # 각 노드가 남기는 실행 로그. 노드 업데이트마다 기존 로그 뒤에 추가된다.
    analysis_log: Annotated[list[str], operator.add]


def create_initial_state(user_query: str, max_iteration: int) -> StrategyState:
    """기술 전략 워크플로우 시작에 사용할 기본 중첩 상태를 만든다."""
    return {
        "user_query": user_query,
        "scope": {
            "subject_company": "",
            "target_technology": "",
            "target_technologies": [],
            "target_competitors": [],
        },
        "query_plan": {},
        "retrieval": {
            "query": "",
            "candidate_docs": [],
            "filtered_docs": [],
            "retrieved_docs": [],
            "relevance_scores": [],
            "confidence": 0.0,
            "is_success": False,
            "failure_reason": "",
            "attempt": 0,
        },
        "web_search": {
            "queries": [],
            "web_results": [],
            "source_diversity": 0,
            "freshness_score": 0.0,
            "source_reliability_score": 0.0,
            "has_counter_evidence": False,
            "bias_risk_score": 1.0,
            "competitor_coverage": {},
            "balanced_company_coverage": False,
            "is_success": False,
            "failure_reason": "",
            "attempt": 0,
        },
        "assessment": {
            "evidence_bundle": {},
            "results": [],
            "is_complete": False,
            "failure_reason": "",
        },
        "decision": {
            "result": {},
            "is_valid": False,
            "failure_reason": "",
        },
        "draft": {
            "markdown_text": "",
            "version": 0,
            "quality_score": 0.0,
            "needs_revision": True,
            "is_valid": False,
            "failure_reason": "",
        },
        "output": {
            "pdf_path": "",
            "final_pdf_path": "",
            "is_pdf_generated": False,
            "format_error": None,
        },
        "control": {
            "status": "running",
            "workflow_stage": "supervisor",
            "coverage_status": "unknown",
            "is_information_sufficient": False,
            "retry_count": 0,
            "max_iteration": max_iteration,
            "next_step": "supervisor",
            "final_decision": "",
            "query_rewrite_history": [],
        },
        "analysis_log": [],
    }
