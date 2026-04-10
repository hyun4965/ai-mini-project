from __future__ import annotations

from typing import Any, TypedDict

from .state import (
    AssessmentState,
    ControlState,
    DecisionState,
    DraftState,
    OutputState,
    RetrievalState,
    ScopeState,
    WebSearchState,
)


class RouteInput(TypedDict):
    """LangGraph 조건부 라우터가 읽는 최소 상태."""
    control: ControlState


class SupervisorInput(TypedDict):
    """Supervisor가 다음 노드를 고르기 위해 확인하는 상태 그룹."""
    user_query: str
    scope: ScopeState
    query_plan: dict[str, Any]
    retrieval: RetrievalState
    web_search: WebSearchState
    assessment: AssessmentState
    decision: DecisionState
    draft: DraftState
    output: OutputState
    control: ControlState


class SupervisorUpdate(TypedDict, total=False):
    """Supervisor 노드가 반환하는 부분 업데이트."""
    scope: ScopeState
    query_plan: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class RetrievalInput(TypedDict):
    """Retrieval 노드가 실행에 필요로 하는 상태 그룹."""
    scope: ScopeState
    query_plan: dict[str, Any]
    retrieval: RetrievalState
    control: ControlState


class RetrievalUpdate(TypedDict, total=False):
    """Retrieval 노드가 반환하는 부분 업데이트."""
    query_plan: dict[str, Any]
    retrieval: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class WebSearchInput(TypedDict):
    """Web Search 서비스 노드가 실행에 필요로 하는 상태 그룹."""
    scope: ScopeState
    query_plan: dict[str, Any]
    web_search: WebSearchState
    control: ControlState


class WebSearchUpdate(TypedDict, total=False):
    """Web Search 서비스 노드가 반환하는 부분 업데이트."""
    query_plan: dict[str, Any]
    web_search: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class AssessmentInput(TypedDict):
    """근거 묶음과 평가 결과를 만들 때 필요한 상태 그룹."""
    user_query: str
    scope: ScopeState
    retrieval: RetrievalState
    web_search: WebSearchState
    assessment: AssessmentState


class AssessmentUpdate(TypedDict, total=False):
    """Assessment 노드가 반환하는 부분 업데이트."""
    assessment: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class DecisionInput(TypedDict):
    """R&D 추천 결정을 만들 때 필요한 상태 그룹."""
    user_query: str
    scope: ScopeState
    assessment: AssessmentState
    decision: DecisionState


class DecisionUpdate(TypedDict, total=False):
    """Decision 노드가 반환하는 부분 업데이트."""
    decision: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class DraftInput(TypedDict):
    """Markdown 보고서 초안을 작성할 때 필요한 상태 그룹."""
    user_query: str
    scope: ScopeState
    retrieval: RetrievalState
    web_search: WebSearchState
    assessment: AssessmentState
    decision: DecisionState
    draft: DraftState


class DraftUpdate(TypedDict, total=False):
    """Draft 노드가 반환하는 부분 업데이트."""
    draft: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]


class FormattingInput(TypedDict):
    """Markdown을 최종 산출물로 변환할 때 필요한 상태 그룹."""
    draft: DraftState
    output: OutputState


class FormattingUpdate(TypedDict, total=False):
    """Formatting 노드가 반환하는 부분 업데이트."""
    output: dict[str, Any]
    control: dict[str, Any]
    analysis_log: list[str]
