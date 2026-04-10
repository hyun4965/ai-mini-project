from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, BadRequestError, OpenAIError, RateLimitError
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import HashingVectorizer

from .config import StrategyConfig
from .errors import (
    DocumentLoadError,
    EmbeddingInitializationError,
    LLMServiceError,
    OutputWriteError,
    PDFValidationError,
    VectorStoreError,
)
from .formatting import markdown_to_pdf, validate_pdf_output
from .logging_utils import get_logger
from .models import AssessmentResult, DecisionOutput, QueryInterpretation
from .resilience import retry_with_backoff
from .services.web_search import WebSearchService
from .state import StrategyState
from .state_contracts import (
    AssessmentInput,
    AssessmentUpdate,
    DecisionInput,
    DecisionUpdate,
    DraftInput,
    DraftUpdate,
    FormattingInput,
    FormattingUpdate,
    RetrievalInput,
    RetrievalUpdate,
    RouteInput,
    SupervisorInput,
    SupervisorUpdate,
)


class _HashingEmbeddings(Embeddings):
    """HuggingFace 임베딩을 쓸 수 없을 때 FAISS 캐시를 유지하기 위한 로컬 fallback 임베딩."""

    def __init__(self, n_features: int = 2048) -> None:
        """고정 차원 hashing vectorizer를 초기화한다."""
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 목록을 dense float vector 목록으로 변환한다."""
        return self.vectorizer.transform(texts).astype("float32").toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        """단일 query를 dense float vector로 변환한다."""
        return self.embed_documents([text])[0]


QUERY_PROMPT = """당신은 반도체 R&D 전략 워크플로우의 계획 수립 에이전트다.

사용자 시나리오에서 분석 대상 기술과 경쟁사 범위를 추출하라.
- 사용자가 HBM4, HBM, PIM, CXL 중 하나 이상을 언급하면 허용된 기술만 남긴다.
- 경쟁사가 명시되지 않으면 시나리오와 관련된 주요 메모리/플랫폼 경쟁사를 사용한다.
- 이 워크플로우의 기준 기업은 SK hynix다. 사용자가 특별히 요구하지 않는 한 target_competitors 안에 SK hynix는 넣지 않는다.
- Retrieval query, 최신 Web query, 반증 query를 함께 생성해 낙관적 해석과 회의적 해석을 모두 검증할 수 있게 한다.

반환은 구조화 데이터만 사용한다.
"""

ASSESSMENT_PROMPT = """당신은 반도체 기술 전략 분석용 Assessment Agent다.

evidence bundle을 사용해 아래를 평가하라.
1. 직접 근거와 간접 지표 구분
2. TRL 1~9 평가
3. 동일 기준 기반 Threat 평가

평가 규칙:
- TRL 1~3: 기초 연구 단계
- TRL 4~6: 시제품, 검증, 파일럿 단계
- TRL 7~9: 제품화, 고객 검증, 양산 단계
- 공개 기사, 로드맵, 특허, 학회 언급, 마케팅 표현은 보수적으로 해석하며 보통 TRL 1~3, 직접 기술 근거가 있어도 최대 TRL 4 수준으로 본다.
- 공식 샘플링, qualification, pilot, shipment, production 근거가 없으면 TRL 5 이상을 주지 않는다.
- 정확한 기술에 대해 공식 양산 또는 고객 출하 근거가 없는 한 공개 정보만으로 TRL 7~9를 부여하지 않는다.
- TRL 4~6 구간은 공개 정보 기반 추정이라는 점과 내부 검증/수율/고객 데이터 부재에 따른 불확실성을 반드시 설명한다.
- Threat는 기술 성숙도, 시장 영향력, 경쟁 강도를 함께 반영한다.
- 아래 질문에 답하도록 요약한다:
  - 현재 공개 정보 기준 기술 수준은 어떠한가?
  - 이 경쟁사는 어느 수준으로 보이는가?
  - SK hynix 관점에서 위협은 얼마나 큰가?
- relative_position_to_sk_hynix 값은 Ahead / Comparable / Behind / Unclear 중 하나를 사용한다.
- commercialization_signal 값은 Research / Prototype / Pilot / Production / Unclear 중 하나를 사용한다.

중요:
- 서술형 필드(근거 설명, 요약, rationale, implication, uncertainty_note)는 반드시 한국어로 작성한다.
- 기술명, 회사명, TRL, Threat, Go/Hold/Monitor, High/Medium/Low, enum 값은 필요한 경우 원문 표기를 유지해도 된다.

반환은 구조화 데이터만 사용한다.
"""

DECISION_PROMPT = """당신은 SK hynix 기술 전략 Decision Agent다.

assessment 결과를 바탕으로 기술별로 다음을 생성하라.
- Go / Hold / Monitor 권고
- High / Medium / Low 우선순위
- 근거가 분명한 판단 rationale
- 즉시 대응이 필요할 때의 실행 제안

규칙:
- 판단은 반드시 TRL, Threat, 경쟁사 상대 위치, 근거 품질에 기반해야 한다.
- 단순 요약이 아니라 실제 전략적 선택이 드러나야 한다.
- summary, portfolio_view, decision_rationale, suggested_actions는 반드시 한국어로 작성한다.
- enum 값(Go / Hold / Monitor, High / Medium / Low)은 그대로 사용해도 된다.
"""

DRAFT_PROMPT = """당신은 R&D 의사결정 지원용 기술 전략 분석 보고서를 작성하는 Draft Agent다.

아래 제목을 정확히 사용하라.
- # SUMMARY
- ## 1. 분석 배경
- ### 1.1 분석 목적
- ### 1.2 분석 범위 및 기준
- ### 1.3 TRL 기반 평가 기준 정의
- ## 2. 분석 대상 기술 현황
- ### 2.1 HBM 기술 현황
- ### 2.2 PIM 기술 현황
- ### 2.3 CXL 기술 현황
- ## 3. 경쟁사 동향 분석
- ### 3.1 경쟁사별 기술 개발 방향
- ### 3.2 TRL 기반 기술 성숙도 비교
- ### 3.3 위협 수준 평가
- ## 4. 전략적 시사점
- ### 4.1 기술별 전략적 중요도
- ### 4.2 경쟁 대응 방향
- ### 4.3 한계
- ## REFERENCE

작성 규칙:
- 보고서 전체 서술은 한국어로 작성한다.
- 기술명, 회사명, TRL, Threat, Go/Hold/Monitor, High/Medium/Low 같은 용어는 필요한 경우 원문 표기를 유지해도 된다.
- 결론은 반드시 근거와 연결한다.
- Decision 결과를 분명히 반영한다.
- TRL 4~6 불확실성을 명시한다.
- TRL 4~6은 공개 정보 기반 추정이며, 정확한 판단에는 내부 검증 문서, 공정/수율 데이터, 고객 qualification evidence가 필요하다는 점을 분명히 적는다.
- 특허, 학회 활동, 채용, 투자 같은 간접 지표를 사용했다면 명시한다.
- 단순 나열보다 분석형 문단을 우선한다.
"""


class TechStrategyWorkflow:
    """LangGraph 기반 기술 전략 워크플로우를 구성하고 실행한다."""

    def __init__(self, config: StrategyConfig) -> None:
        """LLM 클라이언트, 서비스, 로컬 검색 캐시를 초기화한다."""
        self.config = config
        self.logger = get_logger("workflow")
        self.planner_llm = ChatOpenAI(
            model=config.planner_model,
            temperature=0,
            timeout=config.openai_timeout_seconds,
            max_retries=0,
        )
        self.analysis_llm = ChatOpenAI(
            model=config.analysis_model,
            temperature=0,
            timeout=config.openai_timeout_seconds,
            max_retries=0,
        )
        self.draft_llm = ChatOpenAI(
            model=config.draft_model,
            temperature=0,
            timeout=config.openai_timeout_seconds,
            max_retries=0,
        )
        self.embeddings = None
        self.web_search_service = WebSearchService(config)
        self._chunk_cache: list[Document] | None = None
        self._embedding_cache: list[list[float]] | None = None
        self._vector_store: FAISS | None = None
        self._hashing_embeddings: _HashingEmbeddings | None = None
        self._vector_embedding_backend = ""

    def build(self):
        """워크플로우 graph를 컴파일하고 노드 간 전이를 연결한다."""
        graph = StateGraph(StrategyState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("retrieval", self.retrieval_node)
        graph.add_node("web_search", self.web_search_service.run)
        graph.add_node("assessment", self.assessment_node)
        graph.add_node("decision", self.decision_node)
        graph.add_node("draft", self.draft_node)
        graph.add_node("formatting", self.formatting_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "retrieval": "retrieval",
                "web_search": "web_search",
                "assessment": "assessment",
                "decision": "decision",
                "draft": "draft",
                "formatting": "formatting",
                "END": END,
            },
        )
        for node_name in ("retrieval", "web_search", "assessment", "decision", "draft", "formatting"):
            graph.add_edge(node_name, "supervisor")

        return graph.compile()

    @staticmethod
    def _control_update(
        stage: str,
        *,
        control: dict[str, Any] | None = None,
        rewrite_history: list[str] | None = None,
        **updates: Any,
    ) -> dict[str, Any]:
        """노드 전이에 필요한 제어 상태 부분 업데이트를 만든다."""
        update = {"workflow_stage": stage, **updates}
        if rewrite_history:
            update["query_rewrite_history"] = [
                *(control or {}).get("query_rewrite_history", []),
                *rewrite_history,
            ]
        return update

    def _invoke_llm_with_retry(self, operation_name: str, operation):
        """LLM 호출을 retry/backoff 정책으로 감싼다."""

        def wrapped():
            try:
                return operation()
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                raise LLMServiceError("openai", f"{operation_name} failed: {exc}", retryable=True) from exc
            except APIStatusError as exc:
                retryable = getattr(exc, "status_code", 0) >= 500
                raise LLMServiceError("openai", f"{operation_name} failed with API status error: {exc}", retryable=retryable) from exc
            except (AuthenticationError, BadRequestError, OpenAIError, ValidationError) as exc:
                raise LLMServiceError("openai", f"{operation_name} failed: {exc}", retryable=False) from exc

        return retry_with_backoff(
            wrapped,
            operation_name=operation_name,
            max_retries=self.config.external_api_max_retries,
            base_delay_seconds=self.config.retry_backoff_base_seconds,
            max_delay_seconds=self.config.retry_backoff_max_seconds,
            logger=self.logger,
        )

    def supervisor_node(self, state: SupervisorInput) -> SupervisorUpdate:
        """워크플로우 진행 상황을 점검하고 다음에 필요한 노드로 라우팅한다."""
        interpretation = self._ensure_query_plan(state)
        review = self._compute_review(state, interpretation)
        log_message = review.pop("log_message")
        self.logger.info(
            "supervisor.route next=%s retry=%s/%s coverage=%s",
            review.get("next_step"),
            review.get("retry_count"),
            state["control"].get("max_iteration"),
            review.get("coverage_status"),
        )

        return {
            "scope": {
                "subject_company": self.config.subject_company,
                "target_technology": interpretation["primary_technology"],
                "target_technologies": interpretation["target_technologies"],
                "target_competitors": interpretation["target_competitors"],
            },
            "query_plan": interpretation,
            "control": self._control_update("supervisor", **review),
            "analysis_log": [log_message],
        }

    def retrieval_node(self, state: RetrievalInput) -> RetrievalUpdate:
        """로컬 근거를 검색/필터링하고 실패 시 질의를 다시 작성한다."""
        current_plan = dict(state["query_plan"])
        queries = current_plan.get("retrieval_queries", [])
        self.logger.info("retrieval.start queries=%d", len(queries))
        candidates: list[dict[str, Any]] = []
        threshold_filtered: list[dict[str, Any]] = []
        scores: list[float] = []
        keyword_filtered: list[dict[str, Any]] = []

        for query in queries:
            results = self._retrieve_documents(query)
            for item in results:
                candidates.append(item)
                scores.append(item["relevance_score"])
                if item["relevance_score"] >= self.config.retrieval_score_threshold:
                    threshold_filtered.append(item)

        threshold_filtered = self._dedupe_records(
            threshold_filtered,
            unique_key=lambda item: (item["source"], item["title"], item["content"][:120]),
        )

        for item in threshold_filtered:
            if self._matches_retrieval_scope(item, state):
                keyword_filtered.append(item)

        retrieval_confidence = (
            round(sum(item["relevance_score"] for item in keyword_filtered) / len(keyword_filtered), 4)
            if keyword_filtered
            else 0.0
        )
        is_success = len(keyword_filtered) >= self.config.min_retrieved_docs

        failure_reason = ""
        rewrite_history: list[str] = []
        attempt = state["retrieval"].get("attempt", 0)
        if not is_success:
            failure_reason = self._detect_retrieval_failure_reason(
                candidate_docs=candidates,
                filtered_docs=keyword_filtered,
                relevance_scores=scores,
            )
            current_plan["retrieval_queries"] = self._rewrite_retrieval_queries(
                interpretation=current_plan,
                reason=failure_reason,
            )
            attempt += 1
            rewrite_history = [
                f"[rewrite][retrieval] reason={failure_reason} queries={current_plan['retrieval_queries']}"
            ]
            self.logger.warning("retrieval.retry reason=%s attempt=%d", failure_reason, attempt)
        else:
            self.logger.info("retrieval.complete docs=%d confidence=%.2f", len(keyword_filtered), retrieval_confidence)

        return {
            "query_plan": current_plan,
            "retrieval": {
                "query": " | ".join(queries),
                "candidate_docs": candidates,
                "filtered_docs": keyword_filtered,
                "retrieved_docs": keyword_filtered,
                "relevance_scores": scores,
                "confidence": retrieval_confidence,
                "is_success": is_success,
                "failure_reason": failure_reason,
                "attempt": attempt,
            },
            "control": self._control_update(
                "retrieval",
                control=state["control"],
                rewrite_history=rewrite_history,
            ),
            "analysis_log": [
                f"[retrieval] queries={len(queries)} threshold_docs={len(threshold_filtered)} "
                f"keyword_docs={len(keyword_filtered)} threshold={self.config.retrieval_score_threshold:.2f} "
                f"confidence={retrieval_confidence:.2f}"
            ],
        }

    def assessment_node(self, state: AssessmentInput) -> AssessmentUpdate:
        """근거를 종합해 기술-경쟁사 쌍별 TRL/위협도 평가를 만든다."""
        self.logger.info("assessment.start technologies=%d competitors=%d", len(state["scope"]["target_technologies"] or [state["scope"]["target_technology"]]), len(state["scope"]["target_competitors"]))
        bundle = self._build_evidence_bundle(state)
        assessment_results: list[dict[str, Any]] = []

        scope = state["scope"]
        technologies = scope["target_technologies"] or [scope["target_technology"]]
        competitors = scope["target_competitors"]
        for technology in technologies:
            for competitor in competitors:
                evidence = bundle.get(technology, {}).get(competitor, {"direct_evidence": [], "indirect_evidence": [], "sources": []})
                assessment = self._assess_pair(
                    user_query=state["user_query"],
                    technology=technology,
                    competitor=competitor,
                    evidence=evidence,
                )
                assessment_results.append(assessment)

        analysis_complete, analysis_reason, coverage_status = self._validate_assessment_quality(
            state
            | {
                "assessment": {
                    **state["assessment"],
                    "evidence_bundle": bundle,
                    "results": assessment_results,
                },
            }
        )

        return {
            "assessment": {
                "evidence_bundle": bundle,
                "results": assessment_results,
                "is_complete": analysis_complete,
                "failure_reason": analysis_reason,
            },
            "control": self._control_update(
                "assessment",
                coverage_status=coverage_status,
            ),
            "analysis_log": [
                f"[assessment] assessed_pairs={len(assessment_results)} "
                f"analysis_complete={analysis_complete} reason={analysis_reason or 'ok'}"
            ],
        }

    def decision_node(self, state: DecisionInput) -> DecisionUpdate:
        """평가 결과를 바탕으로 R&D 추천을 생성하고 검증한다."""
        self.logger.info("decision.start assessment_rows=%d", len(state["assessment"].get("results", [])))
        decision = self._make_decision(state)
        is_valid, decision_reason, decision_status = self._validate_decision_quality(
            state | {"decision": {**state["decision"], "result": decision}}
        )
        self.logger.info("decision.complete valid=%s reason=%s", is_valid, decision_reason or "ok")
        return {
            "decision": {
                "result": decision,
                "is_valid": is_valid,
                "failure_reason": decision_reason,
            },
            "control": self._control_update("decision"),
            "analysis_log": [
                f"[decision] recommendations={len(decision.get('recommendations', []))} "
                f"decision_valid={is_valid} reason={decision_reason or 'ok'} status={decision_status}"
            ],
        }

    def draft_node(self, state: DraftInput) -> DraftUpdate:
        """Markdown 보고서 초안을 작성하고 필요하면 규칙 기반 초안으로 대체한다."""
        draft_version = state["draft"]["version"] + 1
        self.logger.info("draft.start version=%d", draft_version)
        markdown = self._draft_report(state)
        quality = self._score_draft(markdown)
        is_valid, draft_reason, draft_status = self._validate_draft_quality(
            state | {"draft": {**state["draft"], "markdown_text": markdown}}
        )
        if not is_valid:
            fallback_markdown = self._build_fallback_draft(state)
            fallback_quality = self._score_draft(fallback_markdown)
            fallback_valid, fallback_reason, fallback_status = self._validate_draft_quality(
                state | {"draft": {**state["draft"], "markdown_text": fallback_markdown}}
            )
            if fallback_quality >= quality:
                markdown = fallback_markdown
                quality = fallback_quality
                is_valid = fallback_valid
                draft_reason = fallback_reason
                draft_status = f"fallback:{fallback_status}"
        self.logger.info("draft.complete version=%d valid=%s quality=%.2f", draft_version, is_valid, quality)
        return {
            "draft": {
                "markdown_text": markdown,
                "version": draft_version,
                "quality_score": quality,
                "needs_revision": not is_valid,
                "is_valid": is_valid,
                "failure_reason": draft_reason,
            },
            "control": self._control_update("draft"),
            "analysis_log": [
                f"[draft] version={draft_version} quality={quality:.2f} "
                f"draft_valid={is_valid} reason={draft_reason or 'ok'} status={draft_status}"
            ],
        }

    def formatting_node(self, state: FormattingInput) -> FormattingUpdate:
        """Markdown 산출물을 저장하고 최종 PDF 산출물로 렌더링한다."""
        label = self.config.deliverable_label
        markdown_path = self.config.output_dir / f"ai-mini_output_{label}.md"
        pdf_path = self.config.output_dir / f"ai-mini_output_{label}.pdf"

        markdown_text = state["draft"]["markdown_text"]
        self.logger.info("formatting.start markdown=%s pdf=%s", markdown_path, pdf_path)
        try:
            try:
                markdown_path.write_text(markdown_text, encoding="utf-8")
            except OSError as exc:
                raise OutputWriteError(f"failed to write markdown output: {exc}") from exc
            generated_path = markdown_to_pdf(markdown_text, pdf_path)
            is_valid_pdf, validation_message = validate_pdf_output(markdown_text, generated_path)
            if not is_valid_pdf:
                raise PDFValidationError(validation_message)
            self.logger.info("formatting.complete pdf=%s", generated_path)
            return {
                "output": {
                    "pdf_path": generated_path,
                    "final_pdf_path": generated_path,
                    "is_pdf_generated": True,
                    "format_error": None,
                },
                "control": self._control_update("formatting", status="completed"),
                "analysis_log": [f"[formatting] pdf={generated_path} validation={validation_message}"],
            }
        except (OSError, RuntimeError, ValueError, OutputWriteError, PDFValidationError) as exc:
            self.logger.exception("formatting.failed: %s", exc)
            return {
                "output": {
                    "pdf_path": "",
                    "final_pdf_path": "",
                    "is_pdf_generated": False,
                    "format_error": str(exc),
                },
                "control": self._control_update("formatting", status="failed"),
                "analysis_log": [f"[formatting] failed={exc}"],
            }

    def _route_supervisor(self, state: RouteInput) -> str:
        """Supervisor가 선택한 다음 graph 노드를 반환한다."""
        return state["control"]["next_step"]

    def _ensure_query_plan(self, state: SupervisorInput) -> dict[str, Any]:
        """기존 질의 계획을 반환하거나 사용자 질의에서 새로 만든다."""
        if state.get("query_plan"):
            return state["query_plan"]

        user_query = state["user_query"]
        lowered_query = user_query.lower()
        subject_company = self.config.subject_company
        explicit_competitors = [
            name
            for name in self.config.competitor_catalog
            if name.lower() in lowered_query and name != subject_company
        ]
        technologies = ", ".join(self.config.technology_catalog)
        competitors = ", ".join(self.config.competitor_catalog)
        parser = self.planner_llm.with_structured_output(QueryInterpretation)
        try:
            result = self._invoke_llm_with_retry(
                "query_planning",
                lambda: parser.invoke(
                    [
                        SystemMessage(content=QUERY_PROMPT),
                        HumanMessage(
                            content=(
                                f"Allowed technologies: {technologies}\n"
                                f"Known competitors: {competitors}\n"
                                f"User scenario: {user_query}"
                            )
                        ),
                    ]
                ),
            )
            interpretation = result.model_dump()
        except LLMServiceError as exc:
            self.logger.warning("query planning fallback triggered: %s", exc)
            interpretation = self._fallback_query_interpretation(state["user_query"])

        if not interpretation["target_technologies"]:
            interpretation["target_technologies"] = ["HBM4", "PIM", "CXL"]
        if not interpretation["primary_technology"]:
            interpretation["primary_technology"] = interpretation["target_technologies"][0]
        if explicit_competitors:
            interpretation["target_competitors"] = explicit_competitors
        else:
            interpretation["target_competitors"] = self._default_target_competitors()
        interpretation["target_competitors"] = [
            competitor for competitor in interpretation["target_competitors"]
            if competitor != subject_company
        ] or self._default_target_competitors()
        return interpretation

    def _fallback_query_interpretation(self, user_query: str) -> dict[str, Any]:
        """LLM 계획 생성이 불가능할 때 규칙 기반 질의 계획을 만든다."""
        lowered = user_query.lower()
        subject_company = self.config.subject_company
        techs = [tech for tech in self.config.technology_catalog if tech.lower() in lowered]
        competitors = [
            name for name in self.config.competitor_catalog
            if name.lower() in lowered and name != subject_company
        ]

        if not techs:
            techs = ["HBM4", "PIM", "CXL"]
        if not competitors:
            competitors = self._default_target_competitors()

        comparison_companies = [subject_company, *competitors]
        retrieval_queries = [f"{tech} {' '.join(comparison_companies)} technology maturity roadmap" for tech in techs]
        web_queries = [f"{tech} {competitor} latest announcement production roadmap 2025 2026" for tech in techs for competitor in competitors[:3]]
        counter_queries = [f"{tech} limitations delays risk adoption challenge" for tech in techs]

        return {
            "primary_technology": techs[0],
            "target_technologies": techs,
            "target_competitors": competitors,
            "reasoning": "키워드 매칭 기반의 규칙형 fallback 질의 해석 결과다.",
            "retrieval_queries": retrieval_queries,
            "web_queries": web_queries[:6],
            "counter_queries": counter_queries[:3],
        }

    def _default_target_competitors(self) -> list[str]:
        """과제 기본 비교군으로 사용할 경쟁사 목록을 반환한다."""
        return [
            competitor
            for competitor in ("Samsung", "Micron")
            if competitor in self.config.competitor_catalog and competitor != self.config.subject_company
        ]

    def _detect_retrieval_failure_reason(
        self,
        *,
        candidate_docs: list[dict[str, Any]],
        filtered_docs: list[dict[str, Any]],
        relevance_scores: list[float],
    ) -> str:
        """로컬 검색이 충분한 근거를 모으지 못한 이유를 분류한다."""
        if len(candidate_docs) == 0:
            return "no_candidates"
        if len(filtered_docs) == 0 and len(candidate_docs) > 0:
            if max(relevance_scores or [0.0], default=0.0) < self.config.retrieval_score_threshold:
                return "low_retrieval_score"
            return "missing_technology_or_competitor_keywords"
        if len(filtered_docs) < self.config.min_retrieved_docs:
            return "insufficient_relevant_docs"
        return "generic_retrieval_failure"

    def _rewrite_retrieval_queries(
        self,
        *,
        interpretation: dict[str, Any],
        reason: str,
    ) -> list[str]:
        """감지된 검색 실패 이유에 맞춰 검색 질의를 확장한다."""
        technologies = interpretation.get("target_technologies", [])
        competitors = interpretation.get("target_competitors", [])
        existing = list(interpretation.get("retrieval_queries", []))
        rewritten: list[str] = []

        for technology in technologies:
            competitor_text = " ".join([self.config.subject_company, *competitors[:3]])
            if reason == "low_retrieval_score":
                rewritten.extend(
                    [
                        f"{technology} {competitor_text} ISSCC Hot Chips JEDEC paper roadmap qualification sample",
                        f"{technology} {competitor_text} conference paper benchmark architecture technical report",
                    ]
                )
            elif reason == "missing_technology_or_competitor_keywords":
                rewritten.extend(
                    [
                        f"{technology} Samsung technical paper",
                        f"{technology} Micron technical paper",
                        f"{technology} {self.config.subject_company} technical paper",
                    ]
                )
            elif reason == "insufficient_relevant_docs":
                rewritten.extend(
                    [
                        f"{technology} {competitor_text} official document whitepaper presentation",
                        f"{technology} {competitor_text} roadmap production sample benchmark",
                    ]
                )
            else:
                rewritten.append(f"{technology} {competitor_text} technology maturity roadmap technical paper")

        return self._dedupe_strings(existing + rewritten)

    def _compute_review(self, state: SupervisorInput, interpretation: dict[str, Any]) -> dict[str, Any]:
        """워크플로우 검증 기준을 평가하고 Supervisor 라우팅 업데이트를 계산한다."""
        info_ok, coverage_status = self._validate_information_sufficiency(state)
        analysis_ok, analysis_reason, analysis_status = self._validate_assessment_quality(state)
        decision_ok, decision_reason, decision_status = self._validate_decision_quality(state)
        draft_ok, draft_reason, draft_status = self._validate_draft_quality(state)

        control = state["control"]
        retrieval = state["retrieval"]
        web_search = state["web_search"]
        output = state["output"]

        retry_count = control["retry_count"]
        next_step = "retrieval"
        final_decision = ""
        status = control["status"]

        if output.get("is_pdf_generated") and output.get("final_pdf_path"):
            next_step = "END"
            final_decision = "END"
            status = "completed"
        elif not retrieval.get("is_success"):
            if control.get("workflow_stage") == "retrieval":
                retry_count += 1
            next_step = "retrieval"
        elif not web_search.get("is_success"):
            if control.get("workflow_stage") == "web_search":
                retry_count += 1
            next_step = "web_search"
        elif not info_ok and retry_count < control["max_iteration"]:
            next_step = "web_search" if web_search.get("source_diversity", 0) < self.config.min_source_diversity or not web_search.get("has_counter_evidence") else "retrieval"
            retry_count += 1
        elif info_ok and not state["assessment"].get("results"):
            next_step = "assessment"
        elif not analysis_ok:
            if control.get("workflow_stage") == "assessment":
                retry_count += 1
            next_step = self._route_analysis_retry(state, analysis_reason)
        elif not decision_ok:
            if control.get("workflow_stage") == "decision":
                retry_count += 1
            next_step = self._route_decision_retry(state, decision_reason)
        elif not draft_ok:
            if control.get("workflow_stage") == "draft":
                retry_count += 1
            next_step = "draft"
        elif not output.get("is_pdf_generated"):
            if control.get("workflow_stage") == "formatting" and output.get("format_error"):
                retry_count += 1
            next_step = "formatting"
        else:
            next_step = "END"
            final_decision = "END"
            status = "completed"

        if retry_count >= control["max_iteration"] and (not info_ok or not analysis_ok or not decision_ok or not draft_ok or output.get("format_error")):
            status = "failed"
            next_step = "END"
            final_decision = "END"

        return {
            "is_information_sufficient": info_ok,
            "coverage_status": coverage_status,
            "retry_count": retry_count,
            "next_step": next_step,
            "final_decision": final_decision,
            "status": status,
            "log_message": (
                f"[supervisor] next={next_step} info={info_ok} analysis={analysis_ok} "
                f"analysis_reason={analysis_reason or 'ok'} analysis_status={analysis_status} "
                f"decision={decision_ok} decision_reason={decision_reason or 'ok'} "
                f"decision_status={decision_status} draft={draft_ok} draft_reason={draft_reason or 'ok'} "
                f"draft_status={draft_status} retry={retry_count}/{control['max_iteration']}"
            ),
        }

    def _validate_information_sufficiency(self, state: SupervisorInput) -> tuple[bool, str]:
        """검색/웹 근거가 최소 커버리지 기준을 만족하는지 확인한다."""
        retrieval = state["retrieval"]
        web_search = state["web_search"]
        filtered_docs = retrieval.get("filtered_docs", [])
        web_results = web_search.get("web_results", [])
        retrieval_ok = len(filtered_docs) >= self.config.min_retrieved_docs and retrieval.get("confidence", 0.0) > 0
        web_ok = (
            len(web_results) >= self.config.min_web_results
            and web_search.get("source_diversity", 0) >= self.config.min_source_diversity
            and web_search.get("freshness_score", 0.0) >= self.config.min_recent_ratio
            and web_search.get("source_reliability_score", 0.0) >= self.config.min_source_reliability_score
            and web_search.get("has_counter_evidence", False)
            and web_search.get("bias_risk_score", 1.0) <= self.config.max_bias_risk_score
            and web_search.get("balanced_company_coverage", False)
        )
        status = (
            f"retrieval:{len(filtered_docs)}/{self.config.min_retrieved_docs}, "
            f"web:{len(web_results)}/{self.config.min_web_results}, "
            f"sources:{web_search.get('source_diversity', 0)}/{self.config.min_source_diversity}, "
            f"recent:{web_search.get('freshness_score', 0.0):.2f}/{self.config.min_recent_ratio:.2f}, "
            f"reliability:{web_search.get('source_reliability_score', 0.0):.2f}/{self.config.min_source_reliability_score:.2f}, "
            f"bias:{web_search.get('bias_risk_score', 1.0):.2f}/{self.config.max_bias_risk_score:.2f}"
        )
        return retrieval_ok and web_ok, status

    def _validate_analysis_complete(self, state: AssessmentInput | SupervisorInput) -> bool:
        """예상 평가 쌍이 모두 품질 검증을 통과했는지 반환한다."""
        analysis_ok, _, _ = self._validate_assessment_quality(state)
        return analysis_ok

    def _validate_assessment_quality(self, state: AssessmentInput | SupervisorInput) -> tuple[bool, str, str]:
        """평가 커버리지, 근거 품질, 지표 범위를 검증한다."""
        scope = state["scope"]
        assessment = state["assessment"]
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        competitors = scope.get("target_competitors", [])
        expected = len(technologies) * len(competitors)
        results = assessment.get("results", [])
        evidence_bundle = assessment.get("evidence_bundle", {})

        if expected == 0:
            return False, "missing_scope", "no technology or competitor scope"
        if len(results) < expected:
            return False, "missing_pairs", f"assessment pairs {len(results)}/{expected}"

        result_index = {
            (item.get("technology"), item.get("competitor")): item
            for item in results
        }

        for technology in technologies:
            for competitor in competitors:
                item = result_index.get((technology, competitor))
                if item is None:
                    return False, "missing_pairs", f"missing assessment for {technology}/{competitor}"

                evidence = evidence_bundle.get(technology, {}).get(
                    competitor,
                    {"direct_evidence": [], "indirect_evidence": [], "sources": []},
                )
                direct = item.get("direct_evidence") or evidence.get("direct_evidence", [])
                indirect = item.get("indirect_evidence") or evidence.get("indirect_evidence", [])
                sources = evidence.get("sources", [])
                total_evidence = len(direct) + len(indirect)

                if total_evidence < 1 or not sources:
                    return (
                        False,
                        "insufficient_evidence",
                        f"{technology}/{competitor} evidence total={total_evidence} sources={len(sources)}",
                    )
                if not direct:
                    return False, "missing_direct_evidence", f"{technology}/{competitor} has no direct evidence"

                trl_level = item.get("trl_level")
                if not isinstance(trl_level, int) or not 1 <= trl_level <= 9:
                    return False, "invalid_trl_range", f"{technology}/{competitor} trl={trl_level}"
                if not item.get("trl_rationale"):
                    return False, "missing_trl_rationale", f"{technology}/{competitor} missing TRL rationale"
                if 4 <= trl_level <= 6 and not item.get("uncertainty_note"):
                    return False, "missing_trl_uncertainty", f"{technology}/{competitor} missing TRL 4-6 uncertainty"

                if not item.get("threat_rationale"):
                    return False, "missing_threat_rationale", f"{technology}/{competitor} missing threat rationale"

                for metric_name in ("threat_score", "market_impact", "competition_intensity"):
                    metric_value = item.get(metric_name)
                    if not isinstance(metric_value, (int, float)) or not 0.0 <= float(metric_value) <= 1.0:
                        return False, "non_comparable_threat", f"{technology}/{competitor} invalid {metric_name}={metric_value}"

        return True, "", f"assessment pairs validated {expected}/{expected}"

    def _route_analysis_retry(self, state: SupervisorInput, reason: str) -> str:
        """Assessment 품질 검증이 실패했을 때 가장 적절한 재시도 노드를 고른다."""
        retrieval = state["retrieval"]
        web_search = state["web_search"]
        assessment = state["assessment"]
        if not assessment.get("results"):
            return "assessment"
        if reason in {"missing_pairs", "insufficient_evidence", "missing_direct_evidence"}:
            if len(retrieval.get("filtered_docs", [])) < self.config.min_retrieved_docs:
                return "retrieval"
            return "web_search"

        if reason in {"missing_trl_uncertainty", "invalid_trl_range", "missing_trl_rationale"}:
            if web_search.get("source_diversity", 0) < self.config.min_source_diversity or web_search.get("freshness_score", 0.0) < self.config.min_recent_ratio:
                return "web_search"
            return "retrieval"

        if reason in {"missing_threat_rationale", "non_comparable_threat"}:
            return "web_search" if not web_search.get("has_counter_evidence") else "retrieval"

        return "assessment"

    def _validate_decision(self, state: DecisionInput | SupervisorInput) -> bool:
        """의사결정 결과가 품질 검증을 통과했는지 반환한다."""
        decision_ok, _, _ = self._validate_decision_quality(state)
        return decision_ok

    def _validate_decision_quality(self, state: DecisionInput | SupervisorInput) -> tuple[bool, str, str]:
        """추천 커버리지, 형식, 근거 연결성, 실행안을 검증한다."""
        scope = state["scope"]
        assessment = state["assessment"]
        decision = state["decision"]
        if not assessment.get("is_complete") or not assessment.get("results"):
            return False, "insufficient_assessment", "analysis is incomplete"

        decision_result = decision.get("result", {})
        recommendations = decision_result.get("recommendations", [])
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        assessments = assessment.get("results", [])

        if not technologies:
            return False, "missing_scope", "no technology scope for decision"
        if not isinstance(decision_result.get("summary"), str) or not decision_result.get("summary", "").strip():
            return False, "missing_decision_summary", "decision summary missing"
        if not isinstance(decision_result.get("portfolio_view"), str) or not decision_result.get("portfolio_view", "").strip():
            return False, "missing_portfolio_view", "portfolio view missing"
        if len(recommendations) < len(technologies):
            return False, "missing_recommendations", f"recommendations {len(recommendations)}/{len(technologies)}"

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in assessments:
            grouped[item["technology"]].append(item)

        recommendation_index = {
            item.get("technology"): item
            for item in recommendations
            if item.get("technology")
        }

        for technology in technologies:
            rec = recommendation_index.get(technology)
            if rec is None:
                return False, "missing_recommendations", f"missing recommendation for {technology}"
            if rec.get("rd_feasibility") not in {"Go", "Hold", "Monitor"}:
                return False, "invalid_decision_format", f"{technology} invalid rd_feasibility"
            if rec.get("priority_level") not in {"High", "Medium", "Low"}:
                return False, "invalid_decision_format", f"{technology} invalid priority"
            if not isinstance(rec.get("decision_score"), (int, float)) or not 0.0 <= float(rec.get("decision_score")) <= 1.0:
                return False, "invalid_decision_format", f"{technology} invalid decision_score"

            rationale = (rec.get("decision_rationale") or "").strip()
            if not rationale:
                return False, "missing_decision_rationale", f"{technology} rationale missing"

            rationale_lower = rationale.lower()
            grounded_terms = [
                "trl", "threat", "evidence", "competitor",
                "위협", "근거", "경쟁", "성숙", "trl",
            ]
            grounded_hits = sum(1 for token in grounded_terms if token in rationale_lower)
            if grounded_hits < 2:
                return False, "ungrounded_decision_rationale", f"{technology} rationale not grounded enough"

            actions = rec.get("suggested_actions", [])
            if not isinstance(actions, list) or not actions:
                return False, "missing_strategic_actions", f"{technology} suggested actions missing"

            competitors = rec.get("target_competitors", [])
            if not isinstance(competitors, list) or not competitors:
                return False, "missing_competitor_link", f"{technology} target competitors missing"

            rows = grouped.get(technology, [])
            if not rows:
                return False, "insufficient_assessment", f"{technology} has no assessment rows"

        return True, "", f"decision recommendations validated {len(technologies)}/{len(technologies)}"

    def _route_decision_retry(self, state: SupervisorInput, reason: str) -> str:
        """Decision 검증이 실패했을 때 재시도할 노드를 고른다."""
        if reason in {
            "insufficient_assessment",
            "missing_competitor_link",
            "missing_decision_rationale",
            "ungrounded_decision_rationale",
        }:
            return "assessment"

        if reason in {
            "missing_recommendations",
            "invalid_decision_format",
            "missing_decision_summary",
            "missing_portfolio_view",
            "missing_strategic_actions",
        }:
            return "decision"

        return "decision"

    def _validate_draft(self, state: SupervisorInput) -> bool:
        """보고서 초안이 품질 검증을 통과했는지 반환한다."""
        draft_ok, _, _ = self._validate_draft_quality(state)
        return draft_ok

    @staticmethod
    def _required_draft_headings() -> list[str]:
        """최종 보고서가 반드시 포함해야 하는 표준 목차를 반환한다."""
        return [
            "# SUMMARY",
            "## 1. 분석 배경",
            "### 1.1 분석 목적",
            "### 1.2 분석 범위 및 기준",
            "### 1.3 TRL 기반 평가 기준 정의",
            "## 2. 분석 대상 기술 현황",
            "### 2.1 HBM 기술 현황",
            "### 2.2 PIM 기술 현황",
            "### 2.3 CXL 기술 현황",
            "## 3. 경쟁사 동향 분석",
            "### 3.1 경쟁사별 기술 개발 방향",
            "### 3.2 TRL 기반 기술 성숙도 비교",
            "### 3.3 위협 수준 평가",
            "## 4. 전략적 시사점",
            "### 4.1 기술별 전략적 중요도",
            "### 4.2 경쟁 대응 방향",
            "### 4.3 한계",
            "## REFERENCE",
        ]

    def _validate_draft_quality(self, state: DraftInput | SupervisorInput) -> tuple[bool, str, str]:
        """필수 보고서 섹션, 근거 연결성, 분석형 문체를 검증한다."""
        markdown = state["draft"].get("markdown_text", "")
        if not markdown:
            return False, "missing_draft", "draft report is empty"

        required_sections = self._required_draft_headings()
        if not all(section in markdown for section in required_sections):
            return False, "missing_required_sections", "required draft sections missing"

        if not (("TRL 4~6" in markdown or "TRL 4-6" in markdown) and "공개 정보 기반 추정" in markdown and "내부 문서" in markdown):
            return False, "missing_trl_limitation", "TRL 4-6 limitation statement missing"

        decision = state["decision"].get("result", {})
        recommendations = decision.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                technology = rec.get("technology", "")
                feasibility = rec.get("rd_feasibility", "")
                priority = rec.get("priority_level", "")
                if technology and technology not in markdown:
                    return False, "missing_decision_reflection", f"{technology} missing from draft"
                if feasibility and feasibility not in markdown:
                    return False, "missing_decision_reflection", f"{technology} feasibility missing from draft"
                if priority and priority not in markdown:
                    return False, "missing_decision_reflection", f"{technology} priority missing from draft"

        evidence_markers = ("근거", "evidence", "직접 근거", "간접 지표", "출처", "reference")
        if not any(marker.lower() in markdown.lower() for marker in evidence_markers):
            return False, "missing_evidence_linkage", "no evidence linkage markers found"

        assessment_results = state["assessment"].get("results", [])
        competitor_hits = 0
        for item in assessment_results:
            competitor = item.get("competitor", "")
            if competitor and competitor in markdown:
                competitor_hits += 1
        if assessment_results and competitor_hits == 0:
            return False, "missing_competitor_analysis", "competitor analysis not reflected in draft"

        if "TRL" not in markdown or "Threat" not in markdown:
            return False, "missing_assessment_terms", "TRL or Threat discussion missing"

        if self._is_listing_heavy(markdown):
            return False, "listing_heavy_draft", "draft is too list-heavy and not analytic enough"

        return True, "", "draft validated"

    def _load_documents(self) -> list[Document]:
        """로컬 문서 모음을 로드하고 검색용 chunk로 나눈다."""
        if self._chunk_cache is not None:
            return self._chunk_cache

        documents: list[Document] = []
        for path in self.config.data_dir.rglob("*"):
            if path.name.startswith(".") or not path.is_file():
                continue
            normalized_name = path.stem.lower().replace("_", " ")
            if "health belief model" in normalized_name:
                continue
            try:
                if path.suffix.lower() in {".txt", ".md"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                elif path.suffix.lower() == ".pdf":
                    reader = PdfReader(str(path))
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                else:
                    continue
            except (OSError, PdfReadError, ValueError) as exc:
                error = DocumentLoadError(f"failed to load document {path}: {exc}")
                self.logger.warning("document.load.skipped path=%s reason=%s", path, error)
                continue
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": str(path), "title": path.stem}))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        self._chunk_cache = splitter.split_documents(documents) if documents else []
        return self._chunk_cache

    def _get_embeddings(self):
        """임베딩 모델을 지연 초기화하고 실패 시 어휘 기반 점수 계산으로 대체한다."""
        if not self.config.enable_dense_retrieval:
            return None
        if self.embeddings is not None:
            return self.embeddings

        try:
            if self.config.embedding_local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cpu", "local_files_only": self.config.embedding_local_files_only},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            error = EmbeddingInitializationError(
                f"embedding model {self.config.embedding_model} initialization failed: {exc}"
            )
            self.logger.warning("embedding.init.failed model=%s reason=%s", self.config.embedding_model, error)
            self.embeddings = None
        return self.embeddings

    def _get_vector_store_embeddings(self) -> Any | None:
        """FAISS 저장소에 사용할 임베딩을 반환하고 dense 실패 시 hashing fallback을 사용한다."""
        preferred_backend = self.config.embedding_backend.lower()
        saved_backend = self._saved_vector_store_backend()
        if preferred_backend == "hashing" or (preferred_backend == "auto" and saved_backend == "hashing"):
            if self._hashing_embeddings is None:
                self._hashing_embeddings = _HashingEmbeddings()
            self._vector_embedding_backend = "hashing"
            return self._hashing_embeddings

        embeddings = self._get_embeddings()
        if embeddings is not None:
            self._vector_embedding_backend = "huggingface"
            return embeddings

        if not self.config.enable_vector_store:
            return None
        if self._hashing_embeddings is None:
            self._hashing_embeddings = _HashingEmbeddings()
        self._vector_embedding_backend = "hashing"
        return self._hashing_embeddings

    def _saved_vector_store_backend(self) -> str:
        """저장된 FAISS metadata에서 사용된 임베딩 backend 이름을 읽는다."""
        metadata_path = self.config.vector_store_dir / "metadata.json"
        if not metadata_path.exists():
            return ""
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        backend = metadata.get("vector_backend")
        return backend if isinstance(backend, str) else ""

    def _corpus_signature(self, backend_name: str) -> str:
        """문서/청크/임베딩 설정이 바뀌었는지 판별할 fingerprint를 만든다."""
        payload: list[dict[str, Any]] = [
            {
                "embedding_model": self.config.embedding_model,
                "vector_backend": backend_name,
                "chunk_size": 1200,
                "chunk_overlap": 200,
            }
        ]
        for path in sorted(self.config.data_dir.rglob("*")):
            if path.name.startswith(".") or not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                continue
            stat = path.stat()
            payload.append(
                {
                    "path": str(path.relative_to(self.config.project_root)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_or_build_vector_store(
        self,
        chunks: list[Document] | None,
        embeddings: Any,
        backend_name: str,
    ) -> FAISS | None:
        """FAISS 인덱스를 재사용하거나 문서 변경 시 새로 생성해 로컬에 저장한다."""
        if not self.config.enable_vector_store:
            return None
        if self._vector_store is not None:
            return self._vector_store

        index_dir = self.config.vector_store_dir
        metadata_path = index_dir / "metadata.json"
        signature = self._corpus_signature(backend_name)

        if metadata_path.exists() and (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("signature") == signature:
                    self._vector_store = FAISS.load_local(
                        str(index_dir),
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    return self._vector_store
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                error = VectorStoreError(f"vector store load failed: {exc}")
                self.logger.warning("vector_store.load.failed backend=%s reason=%s", backend_name, error)
                self._vector_store = None

        if not chunks:
            return None

        try:
            self._vector_store = FAISS.from_documents(chunks, embeddings)
            self._vector_store.save_local(str(index_dir))
            metadata_path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "embedding_model": self.config.embedding_model,
                        "vector_backend": backend_name,
                        "chunk_count": len(chunks),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return self._vector_store
        except (OSError, ValueError, RuntimeError) as exc:
            error = VectorStoreError(f"vector store build failed: {exc}")
            self.logger.warning("vector_store.build.failed backend=%s reason=%s", backend_name, error)
            self._vector_store = None
            return None

    def _dense_score_from_faiss_distance(self, distance: float) -> float:
        """FAISS L2 거리값을 기존 0~1 dense score 범위로 근사 변환한다."""
        return round(max(0.0, min(1.0, (4.0 - float(distance)) / 4.0)), 4)

    def _retrieve_documents_from_vector_store(
        self,
        query: str,
        chunks: list[Document] | None,
        embeddings: Any,
        backend_name: str,
    ) -> list[dict[str, Any]] | None:
        """저장된 FAISS 벡터 인덱스를 사용해 retrieval 후보를 가져온다."""
        vector_store = self._load_or_build_vector_store(chunks, embeddings, backend_name)
        if vector_store is None:
            return None

        try:
            matches = vector_store.similarity_search_with_score(
                query,
                k=self.config.retrieval_top_k,
                fetch_k=max(self.config.retrieval_top_k * 4, self.config.retrieval_top_k),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            error = VectorStoreError(f"vector store query failed: {exc}")
            self.logger.warning("vector_store.query.failed backend=%s reason=%s", backend_name, error)
            return None

        query_terms = self._tokenize(query)
        scored = []
        for doc, distance in matches:
            content_terms = self._tokenize(doc.page_content)
            lexical_score = len(query_terms & content_terms) / max(len(query_terms), 1)
            dense_score = self._dense_score_from_faiss_distance(distance)
            hybrid_score = round((0.65 * dense_score) + (0.35 * lexical_score), 4)
            scored.append((hybrid_score, lexical_score, dense_score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "title": item.metadata.get("title", Path(item.metadata.get("source", "document")).stem),
                "source": item.metadata.get("source", "local"),
                "source_type": "retrieval",
                "content": item.page_content,
                "relevance_score": round(score, 4),
                "url": None,
                "metadata": {
                    **item.metadata,
                    "retrieval_technique": f"faiss_vector_store_{backend_name}_hybrid",
                    "lexical_score": lexical_score,
                    "dense_score": dense_score,
                },
            }
            for score, lexical_score, dense_score, item in scored[: self.config.retrieval_top_k]
        ]

    def _retrieve_documents(self, query: str) -> list[dict[str, Any]]:
        """dense/lexical 혼합 점수 계산으로 상위 로컬 chunk를 검색한다."""
        embeddings = self._get_vector_store_embeddings()
        if embeddings is not None:
            vector_results = self._retrieve_documents_from_vector_store(
                query,
                None,
                embeddings,
                self._vector_embedding_backend or "unknown",
            )
            if vector_results is not None:
                return vector_results

        chunks = self._load_documents()
        if not chunks:
            return []

        query_terms = self._tokenize(query)
        query_technologies = [
            tech for tech in self.config.technology_catalog if self._contains_alias(query.lower(), self._technology_aliases(tech))
        ]
        query_competitors = [
            competitor
            for competitor in self.config.competitor_catalog
            if self._contains_alias(query.lower(), self._competitor_aliases(competitor))
        ]
        dense_scores: list[float] | None = None

        if embeddings is not None:
            vector_results = self._retrieve_documents_from_vector_store(
                query,
                chunks,
                embeddings,
                self._vector_embedding_backend or "unknown",
            )
            if vector_results is not None:
                return vector_results
            try:
                if self._embedding_cache is None:
                    self._embedding_cache = embeddings.embed_documents([doc.page_content for doc in chunks])
                query_embedding = embeddings.embed_query(query)
                dense_scores = [
                    round((self._cosine_similarity(query_embedding, embedding) + 1) / 2, 4)
                    for embedding in self._embedding_cache
                ]
            except (OSError, ValueError, RuntimeError) as exc:
                self.logger.warning("dense_embedding.score.failed query=%s reason=%s", query[:80], exc)
                dense_scores = None

        scored = []
        for idx, doc in enumerate(chunks):
            content_terms = self._tokenize(doc.page_content)
            lexical_score = len(query_terms & content_terms) / max(len(query_terms), 1)
            dense_score = dense_scores[idx] if dense_scores is not None else 0.0
            haystack = "\n".join(
                [
                    doc.page_content,
                    str(doc.metadata.get("title", "")),
                    str(doc.metadata.get("source", "")),
                ]
            ).lower()
            title_source = "\n".join(
                [
                    str(doc.metadata.get("title", "")),
                    str(doc.metadata.get("source", "")),
                ]
            ).lower()
            technology_score = (
                1.0
                if any(self._contains_alias(haystack, self._technology_aliases(tech)) for tech in query_technologies)
                else 0.0
            )
            competitor_score = (
                1.0
                if any(self._contains_alias(haystack, self._competitor_aliases(competitor)) for competitor in query_competitors)
                else 0.0
            )
            title_bonus = (
                0.1
                if any(self._contains_alias(title_source, self._technology_aliases(tech)) for tech in query_technologies)
                else 0.0
            )
            if dense_scores is not None:
                hybrid_score = round((0.65 * dense_score) + (0.35 * lexical_score), 4)
            else:
                hybrid_score = round(
                    min(
                        1.0,
                        (0.35 * lexical_score)
                        + (0.35 * technology_score)
                        + (0.20 * competitor_score)
                        + title_bonus,
                    ),
                    4,
                )
            scored.append((hybrid_score, lexical_score, dense_score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: self.config.retrieval_top_k]
        return [
            {
                "title": item.metadata.get("title", Path(item.metadata.get("source", "document")).stem),
                "source": item.metadata.get("source", "local"),
                "source_type": "retrieval",
                "content": item.page_content,
                "relevance_score": round(score, 4),
                "url": None,
                "metadata": {
                    **item.metadata,
                    "retrieval_technique": "hybrid_dense_lexical" if dense_scores is not None else "lexical_overlap",
                    "lexical_score": lexical_score,
                    "dense_score": dense_score,
                },
            }
            for score, lexical_score, dense_score, item in top
        ]

    def _matches_retrieval_scope(self, doc: dict[str, Any], state: RetrievalInput) -> bool:
        """검색된 문서가 목표 분석 범위를 언급하는지 확인한다."""
        scope = state["scope"]
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        competitors = scope.get("target_competitors", [])
        haystack = "\n".join(
            [
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("source", ""),
            ]
        ).lower()

        tech_match = any(self._contains_alias(haystack, self._technology_aliases(tech)) for tech in technologies)
        competitor_match = any(self._contains_alias(haystack, self._competitor_aliases(competitor)) for competitor in competitors)
        return tech_match and competitor_match

    def _build_evidence_bundle(self, state: AssessmentInput) -> dict[str, Any]:
        """검색/웹 근거를 기술과 경쟁사 기준으로 묶는다."""
        scope = state["scope"]
        technologies = scope["target_technologies"] or [scope["target_technology"]]
        competitors = scope["target_competitors"]
        bundle: dict[str, Any] = {}

        for technology in technologies:
            bundle[technology] = {}
            for competitor in competitors:
                direct_evidence: list[str] = []
                indirect_evidence: list[str] = []
                sources: list[str] = []

                for item in state["retrieval"].get("retrieved_docs", []) + state["web_search"].get("web_results", []):
                    haystack = f'{item.get("title", "")}\n{item.get("content", "")}'.lower()
                    if not self._contains_alias(haystack, self._technology_aliases(technology)):
                        continue
                    if not self._contains_alias(haystack, self._competitor_aliases(competitor)):
                        continue

                    entry = f'{item.get("title", "Untitled")}: {item.get("content", "")[:500]}'
                    if self._is_direct_evidence(entry):
                        direct_evidence.append(entry)
                    else:
                        indirect_evidence.append(entry)
                    if item.get("url"):
                        sources.append(item["url"])
                    else:
                        sources.append(item.get("source", "unknown"))

                bundle[technology][competitor] = {
                    "direct_evidence": self._dedupe_strings(direct_evidence),
                    "indirect_evidence": self._dedupe_strings(indirect_evidence),
                    "sources": self._dedupe_strings(sources),
                }
        return bundle

    def _assess_pair(self, user_query: str, technology: str, competitor: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """기술/경쟁사 한 쌍을 LLM 또는 휴리스틱 대체 로직으로 평가한다."""
        schema = self.analysis_llm.with_structured_output(AssessmentResult)
        payload = {
            "technology": technology,
            "competitor": competitor,
            "direct_evidence": evidence.get("direct_evidence", []),
            "indirect_evidence": evidence.get("indirect_evidence", []),
            "sources": evidence.get("sources", []),
        }

        try:
            result = self._invoke_llm_with_retry(
                f"assessment:{technology}:{competitor}",
                lambda: schema.invoke(
                    [
                        SystemMessage(content=ASSESSMENT_PROMPT),
                        HumanMessage(
                            content=(
                                f"사용자 질의: {user_query}\n"
                                f"평가 대상: {technology} / {competitor}\n"
                                f"Evidence bundle:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                            )
                        ),
                    ]
                ),
            )
            return self._normalize_assessment_result(
                raw=result.model_dump(),
                technology=technology,
                competitor=competitor,
                evidence=evidence,
            )
        except LLMServiceError as exc:
            self.logger.warning(
                "assessment.fallback technology=%s competitor=%s reason=%s",
                technology,
                competitor,
                exc,
            )
            direct = evidence.get("direct_evidence", [])
            indirect = evidence.get("indirect_evidence", [])
            trl = self._infer_trl_from_evidence(
                direct_count=len(direct),
                indirect_count=len(indirect),
            )
            return self._normalize_assessment_result(
                raw={
                    "technology": technology,
                    "competitor": competitor,
                    "direct_evidence": direct,
                    "indirect_evidence": indirect,
                    "trl_level": trl,
                    "trl_rationale": "수집된 직접 근거와 간접 지표의 양을 기반으로 보수적인 fallback 추정을 수행했다.",
                    "current_status_summary": (
                        f"{competitor}의 {technology}는 공개 정보 기준 TRL {trl} 수준으로 추정된다."
                    ),
                    "competitor_level_summary": (
                        f"{competitor}의 공개 신호를 SK hynix 대비 동일한 TRL·Threat 기준으로 정규화했다."
                    ),
                    "threat_rationale": (
                        "Threat 점수는 TRL, 시장 영향력, 경쟁 강도를 동일 기준으로 정규화해 계산했다."
                    ),
                },
                technology=technology,
                competitor=competitor,
                evidence=evidence,
            )

    def _make_decision(self, state: DecisionInput) -> dict[str, Any]:
        """LLM 또는 휴리스틱 대체 로직으로 포트폴리오 추천을 만든다."""
        schema = self.analysis_llm.with_structured_output(DecisionOutput)
        assessments = state["assessment"].get("results", [])
        try:
            result = self._invoke_llm_with_retry(
                "decision",
                lambda: schema.invoke(
                    [
                        SystemMessage(content=DECISION_PROMPT),
                        HumanMessage(
                            content=(
                                f"사용자 질의: {state['user_query']}\n"
                                f"Assessment 결과:\n{json.dumps(assessments, ensure_ascii=False, indent=2)}"
                            )
                        ),
                    ]
                ),
            )
            return self._normalize_decision_output(result.model_dump(), state)
        except LLMServiceError as exc:
            self.logger.warning("decision.fallback reason=%s", exc)
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in assessments:
                grouped[item["technology"]].append(item)

            recommendations = []
            for technology, items in grouped.items():
                avg_threat = sum(item["threat_score"] for item in items) / len(items)
                avg_trl = sum(item["trl_level"] for item in items) / len(items)
                ahead_count = sum(1 for item in items if item.get("relative_position_to_sk_hynix") == "Ahead")
                priority = "High" if avg_threat >= 0.7 else "Medium" if avg_threat >= 0.45 else "Low"
                if avg_threat >= 0.7 or ahead_count >= 1:
                    feasibility = "Go"
                elif avg_threat >= 0.4:
                    feasibility = "Hold"
                else:
                    feasibility = "Monitor"
                recommendations.append(
                    {
                        "technology": technology,
                        "rd_feasibility": feasibility,
                        "priority_level": priority,
                        "decision_score": round((avg_threat * 0.6) + (avg_trl / 9 * 0.4), 2),
                        "decision_rationale": (
                            f"평균 Threat {avg_threat:.2f}, 평균 TRL {avg_trl:.1f}, "
                            f"그리고 SK hynix 대비 우위로 평가된 경쟁사 수 {ahead_count}를 함께 반영해 fallback 결정을 내렸다."
                        ),
                        "is_action_required": priority == "High",
                        "suggested_actions": [
                            "경쟁사 공개 로드맵과 발표를 집중 모니터링한다.",
                            "핵심 가설 검증과 파트너 검증 범위를 확대한다.",
                        ],
                        "target_competitors": sorted({item["competitor"] for item in items}),
                    }
                )

            recommendations.sort(key=lambda item: {"High": 0, "Medium": 1, "Low": 2}[item["priority_level"]])
            return self._normalize_decision_output({
                "summary": "Assessment 결과를 바탕으로 fallback 포트폴리오 관점을 생성했다.",
                "recommendations": recommendations,
                "portfolio_view": "위협 수준이 높고 성숙도 격차가 의미 있게 나타나는 기술을 우선 관리한다.",
            }, state)

    def _normalize_decision_output(self, raw: dict[str, Any], state: DecisionInput | SupervisorInput) -> dict[str, Any]:
        """모든 목표 기술에 유효한 추천이 생기도록 의사결정 출력을 정규화한다."""
        assessments = state["assessment"].get("results", [])
        scope = state["scope"]
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in assessments:
            grouped[item["technology"]].append(item)

        raw_recommendations = {
            item.get("technology"): item
            for item in raw.get("recommendations", [])
            if item.get("technology")
        }

        normalized_recommendations: list[dict[str, Any]] = []
        for technology in technologies:
            rows = grouped.get(technology, [])
            if not rows:
                continue

            avg_threat = sum(item["threat_score"] for item in rows) / len(rows)
            avg_trl = sum(item["trl_level"] for item in rows) / len(rows)
            avg_evidence_quality = sum(item["evidence_quality_score"] for item in rows) / len(rows)
            ahead_competitors = sorted(
                {item["competitor"] for item in rows if item.get("relative_position_to_sk_hynix") == "Ahead"}
            )
            all_competitors = sorted({item["competitor"] for item in rows})

            default_priority = "High" if avg_threat >= 0.7 else "Medium" if avg_threat >= 0.45 else "Low"
            if avg_threat >= 0.7 or ahead_competitors:
                default_feasibility = "Go"
            elif avg_threat >= 0.4:
                default_feasibility = "Hold"
            else:
                default_feasibility = "Monitor"

            default_score = round((avg_threat * 0.55) + ((avg_trl / 9) * 0.30) + (avg_evidence_quality * 0.15), 2)
            ahead_text = (
                f"{', '.join(ahead_competitors)}가 SK hynix 대비 우위로 평가된 점"
                if ahead_competitors
                else "명확히 우위인 경쟁사가 확인되지 않은 점"
            )
            default_rationale = (
                f"이 판단은 평균 TRL {avg_trl:.1f}, 평균 Threat {avg_threat:.2f}, "
                f"근거 품질 {avg_evidence_quality:.2f}, 그리고 {ahead_text}을 함께 반영한 결과다."
            )
            default_actions = self._build_default_actions(default_feasibility, technology, ahead_competitors)

            raw_item = raw_recommendations.get(technology, {})
            rationale = self._prefer_korean_text(raw_item.get("decision_rationale"), default_rationale)
            priority_level = (
                raw_item.get("priority_level")
                if raw_item.get("priority_level") in {"High", "Medium", "Low"}
                else default_priority
            )
            rd_feasibility = (
                raw_item.get("rd_feasibility")
                if raw_item.get("rd_feasibility") in {"Go", "Hold", "Monitor"}
                else default_feasibility
            )
            raw_score = raw_item.get("decision_score", default_score)
            if not isinstance(raw_score, (int, float)) or not 0.0 <= float(raw_score) <= 1.0:
                raw_score = default_score
            raw_actions = raw_item.get("suggested_actions")
            actions = default_actions
            if isinstance(raw_actions, list) and raw_actions:
                normalized_actions = [item.strip() for item in raw_actions if isinstance(item, str) and item.strip()]
                if normalized_actions and any(self._contains_korean(item) for item in normalized_actions):
                    actions = normalized_actions
            normalized_recommendations.append(
                {
                    "technology": technology,
                    "rd_feasibility": rd_feasibility,
                    "priority_level": priority_level,
                    "decision_score": round(float(raw_score), 2),
                    "decision_rationale": rationale,
                    "is_action_required": bool(raw_item.get("is_action_required"))
                    if isinstance(raw_item.get("is_action_required"), bool)
                    else priority_level == "High",
                    "suggested_actions": actions,
                    "target_competitors": raw_item.get("target_competitors")
                    if isinstance(raw_item.get("target_competitors"), list) and raw_item.get("target_competitors")
                    else all_competitors,
                }
            )

        normalized_recommendations.sort(key=lambda item: {"High": 0, "Medium": 1, "Low": 2}[item["priority_level"]])
        summary = self._prefer_korean_text(
            raw.get("summary"),
            "TRL, Threat, 경쟁사 상대 위치, 근거 품질을 종합해 기술별 R&D 우선순위를 정리했다.",
        )
        portfolio_view = self._prefer_korean_text(
            raw.get("portfolio_view"),
            "위협 수준이 높고 성숙도 신호가 강하며 경쟁사 모멘텀이 큰 기술을 우선 대응 대상으로 본다.",
        )
        return {
            "summary": summary,
            "recommendations": normalized_recommendations,
            "portfolio_view": portfolio_view,
        }

    @staticmethod
    def _build_default_actions(feasibility: str, technology: str, ahead_competitors: list[str]) -> list[str]:
        """정규화된 추천 결과에 대한 기본 전략 실행안을 만든다."""
        if feasibility == "Go":
            lead_text = ", ".join(ahead_competitors) if ahead_competitors else "주요 경쟁사"
            return [
                f"{technology} 검증 로드맵을 {lead_text} 기준으로 앞당긴다.",
                f"{technology} 관련 파트너 및 고객 피드백 루프를 조기에 확보한다.",
            ]
        if feasibility == "Hold":
            return [
                f"{technology}에 대한 표적형 프로토타이핑과 핵심 가설 검증을 지속한다.",
                f"{technology} 관련 경쟁사 이정표와 표준화 활동을 계속 추적한다.",
            ]
        return [
            f"{technology} 관련 공개 발표와 생태계 신호를 지속 모니터링한다.",
            f"{technology} 투자 우선순위를 높이기 전 evidence bundle을 다시 갱신한다.",
        ]

    def _draft_report(self, state: DraftInput) -> str:
        """LLM 생성과 규칙 기반 대체 로직으로 최종 보고서 초안을 만든다."""
        assessments = state["assessment"].get("results", [])
        decision = state["decision"].get("result", {})
        scope = state["scope"]
        references = self._collect_references(state)
        subject_company = scope.get("subject_company") or self.config.subject_company

        try:
            response = self._invoke_llm_with_retry(
                "draft",
                lambda: self.draft_llm.invoke(
                    [
                        SystemMessage(content=DRAFT_PROMPT),
                        HumanMessage(
                            content=(
                                f"사용자 질의: {state['user_query']}\n"
                                f"기준 기업: {subject_company}\n"
                                f"대상 기술: {scope['target_technologies']}\n"
                                f"대상 경쟁사: {scope['target_competitors']}\n"
                                f"Assessment 결과: {json.dumps(assessments, ensure_ascii=False, indent=2)}\n"
                                f"Decision 결과: {json.dumps(decision, ensure_ascii=False, indent=2)}\n"
                                f"참고자료: {json.dumps(references, ensure_ascii=False, indent=2)}"
                            )
                        ),
                    ]
                ),
            )
            text = response.content
            if not all(section in text for section in self._required_draft_headings()):
                raise ValueError("Draft missing required section headings")
            if self._has_excessive_english_narrative(text):
                raise ValueError("Draft contains excessive English narrative")
            return text
        except (LLMServiceError, ValueError) as exc:
            self.logger.warning("draft.fallback reason=%s", exc)
            return self._build_fallback_draft(state)

    def _build_fallback_draft(self, state: DraftInput | SupervisorInput) -> str:
        """LLM 초안 생성이나 검증이 실패했을 때 규칙 기반 보고서를 만든다."""
        assessments = state["assessment"].get("results", [])
        decision = state["decision"].get("result", {})
        scope = state["scope"]
        references = self._collect_references(state)
        subject_company = scope.get("subject_company") or self.config.subject_company

        def rows_for(report_technology: str) -> list[dict[str, Any]]:
            """보고서 표준 기술명에 대응하는 평가 row만 골라낸다."""
            aliases = {
                "HBM": {"HBM", "HBM4"},
                "PIM": {"PIM"},
                "CXL": {"CXL"},
            }
            return [row for row in assessments if row.get("technology") in aliases[report_technology]]

        def build_tech_section(report_technology: str) -> str:
            """기술별 fallback 보고서 섹션 본문을 만든다."""
            rows = rows_for(report_technology)
            if not rows:
                return (
                    f"{report_technology}에 대한 충분한 평가 결과가 아직 확보되지 않았지만, 본 보고서는 해당 기술을 분석 대상에 포함한다. "
                    f"향후 Retrieval과 Web Search가 보강되면 기술 개요, 개발 방향, 경쟁사 위치, 직접 근거와 간접 지표를 기반으로 서술을 구체화해야 한다."
                )

            avg_trl = sum(row["trl_level"] for row in rows) / len(rows)
            direct_total = sum(row.get("direct_evidence_count", 0) for row in rows)
            indirect_total = sum(row.get("indirect_evidence_count", 0) for row in rows)
            top_competitors = ", ".join(sorted({row["competitor"] for row in rows}))
            if report_technology == "HBM":
                overview = (
                    "HBM은 TSV 기반 적층 구조와 넓은 I/O를 통해 AI 가속기와 데이터센터용 시스템에서 필요한 대역폭을 제공하는 핵심 메모리 계열이다. "
                    "이번 분석에서는 HBM4를 포함한 HBM 계열을 함께 해석했다."
                )
            elif report_technology == "PIM":
                overview = (
                    "PIM은 메모리 내부 또는 메모리 인접 영역에서 연산을 수행해 데이터 이동 병목과 전력 비용을 줄이려는 기술이다. "
                    "AI 추천, 벡터 연산, 데이터 집약형 워크로드에서 적용 가능성이 강조된다."
                )
            else:
                overview = (
                    "CXL은 메모리 확장과 자원 풀링을 가능하게 하는 인터커넥트 기술로, 서버 메모리 구조와 데이터센터 아키텍처 변화에 직접 연결된다."
                )
            direction = " / ".join(
                sorted(
                    {
                        self._label_commercialization_signal(row.get("commercialization_signal", "Unclear"))
                        for row in rows
                        if row.get("commercialization_signal")
                    }
                )
            )
            status_lines = " ".join(row.get("current_status_summary", "") for row in rows[:3])
            return (
                f"{overview} 공개 근거 기준 평균 TRL은 {avg_trl:.1f} 수준으로 정리되며, 직접 근거 {direct_total}건과 간접 지표 {indirect_total}건이 연결되었다. "
                f"주요 비교 대상은 {top_competitors}이며, 현재 개발 방향은 {direction or '불확실'} 신호로 요약된다. "
                f"{status_lines}"
            )

        competitor_direction_lines = []
        for row in assessments:
            competitor_direction_lines.append(
                f"- {row['competitor']} / {row['technology']}: {row.get('competitor_level_summary', '')} "
                f"직접 근거 {row.get('direct_evidence_count', 0)}건, 간접 지표 {row.get('indirect_evidence_count', 0)}건."
            )

        trl_lines = []
        threat_lines = []
        for row in assessments:
            trl_lines.append(
                f"- {row['technology']} / {row['competitor']}: TRL {row['trl_level']} "
                f"(신뢰도 {row.get('trl_confidence', 0.0):.2f}), 상대 위치 {self._label_relative_position(row.get('relative_position_to_sk_hynix', 'Unclear'))}."
            )
            threat_lines.append(
                f"- {row['technology']} / {row['competitor']}: 위협 수준 {self._label_threat_level(row['threat_level'])} "
                f"(점수 {row['threat_score']:.2f}), 시장 영향력 {row['market_impact']:.2f}, 경쟁 강도 {row['competition_intensity']:.2f}. "
                f"{row.get('threat_rationale', '')}"
            )

        decision_lines = []
        for rec in decision.get("recommendations", []):
            decision_lines.append(
                f"- {rec['technology']}: 권고 {rec['rd_feasibility']} / 우선순위 {rec['priority_level']} / "
                f"판단 근거: {rec['decision_rationale']}"
            )

        high_priority = [rec["technology"] for rec in decision.get("recommendations", []) if rec.get("priority_level") == "High"]
        medium_priority = [rec["technology"] for rec in decision.get("recommendations", []) if rec.get("priority_level") == "Medium"]
        reference_lines = "\n".join(f"- {ref}" for ref in references[:25]) or "- 수집된 참고문헌이 아직 없습니다."
        tech_scope_text = "HBM, PIM, CXL"
        competitor_scope_text = ", ".join(scope["target_competitors"]) or "Samsung, Micron"

        return f"""# SUMMARY

본 보고서는 {tech_scope_text}에 대한 경쟁사 분석을 바탕으로 {subject_company}의 R&D 우선순위를 판단하기 위해 작성되었다. Retrieval과 Web Search를 통해 수집한 직접 근거와 간접 지표를 함께 엮어 기술 성숙도(TRL), 경쟁 위협 수준(Threat), 전략적 대응 방향을 연결했다.

- 분석 대상 기술의 현재 수준은 공개 자료 기준으로 비교했으며, HBM은 AI 메모리 경쟁의 핵심, PIM은 중장기 차별화 후보, CXL은 시스템 구조 변화와 직결되는 기술로 해석했다.
- 경쟁사별 비교는 TRL과 Threat를 동일 기준으로 정렬해 기준 기업인 {subject_company} 대비 {competitor_scope_text}의 상대 위치를 파악하도록 구성했다.
- R&D 판단은 Go / Hold / Monitor와 우선순위 High / Medium / Low로 정리했고, 현재 우선 투자 후보는 {", ".join(high_priority) if high_priority else '추가 검증이 필요한 상태'}이며, 중간 우선순위 후보는 {", ".join(medium_priority) if medium_priority else '추가 평가 필요'}로 요약된다.
- TRL 4~6 구간은 공개 정보 기반 추정에 해당하며, TRL 4 이상을 정밀 판정하려면 내부 문서, 통합 검증 기록, 공정 및 수율 데이터, 고객 샘플 검증 자료가 필요하다는 한계를 명시한다.

## 1. 분석 배경

### 1.1 분석 목적

AI 서버, GPU 메모리, 데이터센터 메모리 계층 구조가 빠르게 변하면서 특정 기술을 지금 당장 추격할지, 선택적으로 보류할지, 혹은 장기 모니터링 대상으로 둘지 판단할 필요가 커졌다. 따라서 본 분석은 단순 정보 요약이 아니라, 현재 기술 수준은 어떠한가, 경쟁사는 어디 수준인가, 실제 위협은 어느 정도인가, 지금 R&D를 추진해야 하는가, 어떤 기술에 우선 투자해야 하는가라는 질문에 답하도록 설계되었다.

### 1.2 분석 범위 및 기준

- 분석 대상 기술: {tech_scope_text}
- 기준 기업: {subject_company}
- 분석 대상 경쟁사: {competitor_scope_text}
- 활용 데이터 범위: 공개 자료 기반의 논문, 학회 발표, 기업 공식 발표, 기사, 산업 리포트
- 내부 정보 포함 여부: 내부 정보는 포함하지 않음
- 근거 해석 원칙: 직접 근거(논문, 발표, 양산 기사, 샘플 공급 기사)와 간접 지표(특허, 채용, 투자, 생태계 활동)를 구분하여 정리

### 1.3 TRL 기반 평가 기준 정의

TRL은 NASA의 9단계 기술 성숙도 척도를 반도체 기술 분석에 적용한 기준으로, 본 보고서는 상대적 인상보다 절대적 위치를 제시하기 위해 TRL을 채택했다. TRL 1은 기초 원리 관찰, TRL 2는 개념 정립, TRL 3은 개념 검증, TRL 4는 실험실 환경 부품 검증, TRL 5는 유사 실환경 검증, TRL 6은 시스템 시연, TRL 7은 실제 운용 환경 시제품, TRL 8은 양산 적합성 검증 완료, TRL 9는 상용 양산 및 납품 단계로 해석했다. 본 보고서에서는 TRL 1~3을 기초 연구 단계, TRL 4~6을 실험 및 시연 단계, TRL 7~9를 상용화 및 양산 단계로 묶어 사용했다.

## 2. 분석 대상 기술 현황

### 2.1 HBM 기술 현황

{build_tech_section("HBM")}

### 2.2 PIM 기술 현황

{build_tech_section("PIM")}

### 2.3 CXL 기술 현황

{build_tech_section("CXL")}

## 3. 경쟁사 동향 분석

### 3.1 경쟁사별 기술 개발 방향

경쟁사별 기술 개발 방향은 단일 기사 요약이 아니라, 직접 근거와 간접 지표를 합친 evidence bundle 기준으로 정리했다. 공식 발표나 논문처럼 기술 신호가 강한 자료는 직접 근거로, 특허 활동과 채용, 투자, 표준화 활동은 간접 지표로 분리해 읽었다.

{chr(10).join(competitor_direction_lines) or "- 경쟁사별 기술 개발 방향을 기술할 근거가 아직 부족합니다."}

### 3.2 TRL 기반 기술 성숙도 비교

기술별 TRL 수준 비교는 동일한 1~9 기준을 적용해 경쟁사 간 상대 위치를 맞춰 보는 데 목적이 있다. 다만 TRL 4~6 구간은 공개 정보 기반 추정에 해당하며, 발표나 시연만으로는 내부 검증 완료 여부를 확정할 수 없기 때문에 과도한 단정은 피해야 한다.

{chr(10).join(trl_lines) or "- TRL 비교 결과가 아직 없습니다."}

### 3.3 위협 수준 평가

Threat 평가는 기술 완성도 자체만이 아니라 시장 영향력과 경쟁 강도를 함께 반영했다. 즉, 공개 성숙도가 높고 시장 파급력이 큰 기술일수록 SK hynix 입장에서 대응 우선순위가 높아진다.

{chr(10).join(threat_lines) or "- Threat 평가 결과가 아직 없습니다."}

## 4. 전략적 시사점

### 4.1 기술별 전략적 중요도

R&D 우선순위 판단은 TRL, Threat, 경쟁사 상대 위치, 근거 품질을 함께 반영해 수행했다. 따라서 Go / Hold / Monitor 결과는 단순 의견이 아니라, 어떤 기술이 현재 시점에서 바로 투자 대상인지 설명하는 전략 판단으로 읽어야 한다.

{chr(10).join(decision_lines) or "- Decision 결과가 아직 생성되지 않았습니다."}

### 4.2 경쟁 대응 방향

경쟁 대응 방향은 기술별로 다르게 가져갈 필요가 있다. 위협이 높고 경쟁사가 앞서는 기술은 추격 전략과 고객 검증 속도 확보가 중요하고, 상용화 속도보다 구조적 차별화 여지가 큰 기술은 차별화 전략이 적합하다. 또한 CXL과 같이 생태계와 표준이 중요한 영역은 단독 추격만으로 부족할 수 있어 협력 전략까지 함께 검토해야 한다.

        - 추격 전략: 경쟁사가 기준 기업 대비 우위로 평가된 기술은 샘플 검증, 파트너 연동, 일정 단축 중심으로 대응
- 차별화 전략: 공개 성숙도는 낮지만 구조적 잠재력이 큰 기술은 성능/전력/아키텍처 차별화 포인트를 우선 설계
- 협력 전략: 표준, 인터페이스, 시스템 생태계 의존도가 큰 기술은 고객사와 IP/플랫폼 파트너 협력 병행

### 4.3 한계

- TRL 4~6 구간은 공개 정보 기반 추정에 해당함
- TRL 4 이상을 정확히 판단하려면 내부 문서, 통합 검증 기록, 공정 및 수율 데이터, 고객 샘플 검증 자료가 필요하지만 본 workflow는 공개 자료만 사용함
- 수율, 공정, 성능 데이터는 비공개 영역이어서 정확한 평가에 한계 존재
- 특허 출원 패턴, 학회 발표 빈도 변화, 채용 공고 키워드, 투자 및 협력 발표 같은 간접 지표를 함께 사용했음

## REFERENCE

{reference_lines}
"""

    def _score_draft(self, markdown: str) -> float:
        """필수 표시문자를 기준으로 단순 보고서 품질 점수를 계산한다."""
        criteria = 0.0
        sections = self._required_draft_headings()
        criteria += sum(1 for section in sections if section in markdown) / len(sections) * 0.5
        if any(token in markdown for token in ("Go", "Hold", "Monitor", "Priority")):
            criteria += 0.15
        if ("TRL 4~6" in markdown or "TRL 4-6" in markdown) and "공개 정보 기반 추정" in markdown and "내부 문서" in markdown:
            criteria += 0.15
        if any(token in markdown.lower() for token in ("근거", "evidence", "직접 근거", "간접 지표")):
            criteria += 0.1
        if "TRL" in markdown and "Threat" in markdown:
            criteria += 0.05
        if not self._is_listing_heavy(markdown):
            criteria += 0.05
        return min(1.0, round(criteria, 2))

    @staticmethod
    def _contains_korean(text: str) -> bool:
        """문자열에 한글이 포함되어 있는지 확인한다."""
        return bool(re.search(r"[가-힣]", text or ""))

    @classmethod
    def _prefer_korean_text(cls, value: Any, fallback: str) -> str:
        """모델 출력이 한국어가 아니면 한국어 fallback 문장을 사용한다."""
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and cls._contains_korean(normalized):
                return normalized
        return fallback

    @staticmethod
    def _label_commercialization_signal(signal: str) -> str:
        """영문 상용화 단계를 보고서용 한국어 표현으로 변환한다."""
        mapping = {
            "Research": "연구",
            "Prototype": "시제품",
            "Pilot": "파일럿",
            "Production": "양산",
            "Unclear": "불확실",
        }
        return mapping.get(signal, signal or "불확실")

    @staticmethod
    def _label_relative_position(position: str) -> str:
        """상대 위치 enum을 한국어 표현으로 변환한다."""
        mapping = {
            "Ahead": "우위",
            "Comparable": "유사",
            "Behind": "열위",
            "Unclear": "불확실",
        }
        return mapping.get(position, position or "불확실")

    @staticmethod
    def _label_threat_level(level: str) -> str:
        """위협 수준 enum을 한국어 표현과 함께 반환한다."""
        mapping = {
            "High": "High(높음)",
            "Medium": "Medium(중간)",
            "Low": "Low(낮음)",
        }
        return mapping.get(level, level or "Unclear")

    @staticmethod
    def _is_listing_heavy(markdown: str) -> bool:
        """분석 서술보다 글머리표 나열이 과도한 초안인지 감지한다."""
        lines = [line.rstrip() for line in markdown.splitlines()]
        bullet_lines = [
            line for line in lines
            if line.strip().startswith(("- ", "* "))
        ]
        narrative_lines = [
            line for line in lines
            if line.strip()
            and not line.strip().startswith(("#", "- ", "* "))
            and len(line.strip()) >= 40
        ]
        return len(bullet_lines) > max(10, len(narrative_lines) * 3) and len(narrative_lines) < 5

    @staticmethod
    def _has_excessive_english_narrative(markdown: str) -> bool:
        """보고서 본문에서 영어 서술 비중이 과도하면 fallback 대상으로 간주한다."""
        body = markdown.split("## REFERENCE", maxsplit=1)[0]
        hangul_count = len(re.findall(r"[가-힣]", body))
        english_count = len(re.findall(r"[A-Za-z]", body))
        return hangul_count == 0 or english_count > hangul_count

    def _normalize_assessment_result(
        self,
        *,
        raw: dict[str, Any],
        technology: str,
        competitor: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """평가 필드를 정규화하고 비교 가능한 기본 지표값을 채운다."""
        direct = self._dedupe_strings(raw.get("direct_evidence") or evidence.get("direct_evidence", []))
        indirect = self._dedupe_strings(raw.get("indirect_evidence") or evidence.get("indirect_evidence", []))
        sources = self._dedupe_strings(evidence.get("sources", []))
        direct_count = len(direct)
        indirect_count = len(indirect)
        total = direct_count + indirect_count

        trl_level = raw.get("trl_level")
        if not isinstance(trl_level, int) or not 1 <= trl_level <= 9:
            trl_level = self._infer_trl_from_evidence(
                direct_count=direct_count,
                indirect_count=indirect_count,
            )
        trl_ceiling, ceiling_reason = self._public_trl_ceiling(
            direct_evidence=direct,
            indirect_evidence=indirect,
            sources=sources,
        )
        if trl_level > trl_ceiling:
            trl_level = trl_ceiling

        commercialization_signal = raw.get("commercialization_signal") or self._commercialization_signal_from_trl(trl_level)
        if commercialization_signal not in {"Research", "Prototype", "Pilot", "Production", "Unclear"}:
            commercialization_signal = self._commercialization_signal_from_trl(trl_level)
        commercialization_signal = self._commercialization_signal_from_trl(trl_level)

        market_impact = raw.get("market_impact")
        if not isinstance(market_impact, (int, float)) or not 0.0 <= float(market_impact) <= 1.0:
            market_impact = 0.9 if technology == "HBM4" else 0.8 if technology == "CXL" else 0.7
        market_impact = round(float(market_impact), 2)

        competition_intensity = raw.get("competition_intensity")
        if not isinstance(competition_intensity, (int, float)) or not 0.0 <= float(competition_intensity) <= 1.0:
            competition_intensity = 0.95 if competitor == "Samsung" else 0.9 if competitor == "Micron" else 0.7
        competition_intensity = round(float(competition_intensity), 2)

        trl_component = trl_level / 9
        threat_score = round(
            min(
                0.99,
                (0.45 * trl_component) + (0.30 * market_impact) + (0.25 * competition_intensity),
            ),
            2,
        )
        threat_level = "High" if threat_score >= 0.72 else "Medium" if threat_score >= 0.48 else "Low"

        evidence_quality_score = round(
            min(1.0, (0.25 * min(direct_count, 3)) + (0.1 * min(indirect_count, 3)) + (0.1 * min(len(sources), 3))),
            2,
        )
        evidence_completeness = direct_count >= 1 and total >= 2 and len(sources) >= 1
        trl_confidence = round(min(0.75, 0.35 + (0.12 * min(direct_count, 3)) + (0.06 * min(indirect_count, 3))), 2)

        relative_position = raw.get("relative_position_to_sk_hynix")
        if competitor == "SK hynix":
            relative_position = "Comparable"
        elif relative_position not in {"Ahead", "Comparable", "Behind", "Unclear"}:
            if trl_level >= 7 and threat_score >= 0.72:
                relative_position = "Ahead"
            elif trl_level <= 4 and threat_score < 0.48:
                relative_position = "Behind"
            else:
                relative_position = "Comparable"

        default_uncertainty_note = (
            "TRL 4~6 구간은 공개 정보 기반 추정이다. TRL 4 이상을 정밀하게 판단하려면 내부 검증 문서, 공정 및 수율 데이터, "
            "고객 qualification evidence가 필요하지만 본 workflow에서는 사용할 수 없다."
        )
        uncertainty_note = self._prefer_korean_text(raw.get("uncertainty_note"), "")
        if 4 <= trl_level <= 6 and not uncertainty_note:
            uncertainty_note = default_uncertainty_note
        if ceiling_reason and ceiling_reason not in uncertainty_note:
            uncertainty_note = f"{uncertainty_note} {ceiling_reason}".strip()

        evidence_summary_default = (
            f"{technology}에 대해 {competitor} 관련 직접 근거 {direct_count}건과 간접 지표 {indirect_count}건을 확보했다."
        )
        current_status_summary_default = (
            f"{competitor}의 {technology}는 공개 자료 기준 TRL {trl_level} 수준으로 판단되며, "
            f"상용화 신호는 {self._label_commercialization_signal(commercialization_signal)} 단계에 가깝다."
        )
        competitor_level_summary_default = (
            f"{competitor}는 {technology} 기준 SK hynix 대비 {self._label_relative_position(relative_position)} 위치로 해석된다."
        )
        trl_rationale_default = (
            "TRL은 evidence bundle 안의 직접 근거, 간접 지표, 상용화 신호를 종합해 보수적으로 산정했다."
        )
        trl_rationale = self._prefer_korean_text(raw.get("trl_rationale"), trl_rationale_default)
        if ceiling_reason:
            trl_rationale = f"{trl_rationale} 공개 정보 상한 적용: {ceiling_reason}"
        threat_rationale_default = (
            f"Threat는 동일 기준으로 정규화했으며, TRL {trl_level}/9, 시장 영향력 {market_impact:.2f}, "
            f"경쟁 강도 {competition_intensity:.2f}를 함께 반영했다."
        )
        strategic_implication_default = (
            f"{competitor}의 {technology}는 SK hynix 기준 {self._label_threat_level(threat_level)} 위협으로 평가되며, "
            "성숙도와 시장 압력을 함께 고려한 대응 우선순위 설정이 필요하다."
        )
        evidence_summary = self._prefer_korean_text(raw.get("evidence_summary"), evidence_summary_default)
        current_status_summary = self._prefer_korean_text(raw.get("current_status_summary"), current_status_summary_default)
        competitor_level_summary = self._prefer_korean_text(raw.get("competitor_level_summary"), competitor_level_summary_default)
        threat_rationale = self._prefer_korean_text(raw.get("threat_rationale"), threat_rationale_default)
        strategic_implication = self._prefer_korean_text(raw.get("strategic_implication"), strategic_implication_default)

        return AssessmentResult(
            technology=technology,
            competitor=competitor,
            direct_evidence=direct,
            indirect_evidence=indirect,
            direct_evidence_count=direct_count,
            indirect_evidence_count=indirect_count,
            evidence_quality_score=evidence_quality_score,
            evidence_completeness=evidence_completeness,
            evidence_summary=evidence_summary,
            current_status_summary=current_status_summary,
            competitor_level_summary=competitor_level_summary,
            trl_level=trl_level,
            trl_confidence=trl_confidence,
            trl_rationale=trl_rationale,
            commercialization_signal=commercialization_signal,
            relative_position_to_sk_hynix=relative_position,
            threat_level=threat_level,
            threat_score=threat_score,
            market_impact=market_impact,
            competition_intensity=competition_intensity,
            threat_rationale=threat_rationale,
            strategic_implication=strategic_implication,
            uncertainty_note=uncertainty_note,
        ).model_dump()

    @staticmethod
    def _infer_trl_from_evidence(*, direct_count: int, indirect_count: int) -> int:
        """직접/간접 근거 수를 바탕으로 대략적인 TRL 수준을 추정한다."""
        if direct_count >= 1:
            return 4
        if indirect_count >= 2:
            return 3
        return 2

    @staticmethod
    def _public_trl_ceiling(
        *,
        direct_evidence: list[str],
        indirect_evidence: list[str],
        sources: list[str],
    ) -> tuple[int, str]:
        """공개 자료만으로 판정할 수 있는 보수적 TRL 상한을 계산한다."""
        evidence_text = " ".join([*direct_evidence, *indirect_evidence, *sources]).lower()
        official_markers = (
            "samsung.com",
            "skhynix.com",
            "micron.com",
            "jedec.org",
            "cxlconsortium.org",
            "computeexpresslink.org",
            "official",
            "press release",
            "보도자료",
            "공식",
        )
        sampling_markers = (
            "sample",
            "sampling",
            "qualification",
            "customer validation",
            "prototype",
            "demonstration",
            "샘플",
            "검증",
            "시제품",
            "시연",
        )
        production_markers = (
            "mass production",
            "volume production",
            "commercial production",
            "shipping",
            "customer shipment",
            "product launch",
            "launched",
            "양산",
            "출하",
            "상용",
            "출시",
        )
        has_official_signal = any(marker in evidence_text for marker in official_markers)
        has_sampling_signal = any(marker in evidence_text for marker in sampling_markers)
        has_production_signal = any(marker in evidence_text for marker in production_markers)

        if has_official_signal and has_production_signal:
            return 6, "공개 공식 자료에서 양산/출하 신호가 확인되어도 내부 검증 자료가 없으므로 TRL 6을 상한으로 둔다."
        if has_official_signal and has_sampling_signal:
            return 5, "공개 공식 자료에서 샘플/검증 신호가 확인되어도 내부 검증 자료가 없으므로 TRL 5를 상한으로 둔다."
        if direct_evidence:
            return 4, "공개 자료만으로는 실험실 검증 이상의 단계를 확정하기 어려워 TRL 4를 상한으로 둔다."
        return 3, "직접 검증 근거가 부족해 공개 간접 지표 기준 TRL 3을 상한으로 둔다."

    @staticmethod
    def _commercialization_signal_from_trl(trl_level: int) -> str:
        """TRL 수준을 대략적인 상용화 단계로 매핑한다."""
        if trl_level <= 3:
            return "Research"
        if trl_level <= 5:
            return "Prototype"
        if trl_level == 6:
            return "Pilot"
        return "Production"

    def _collect_references(self, state: DraftInput | SupervisorInput) -> list[str]:
        """보고서에 사용할 로컬/웹 참고 출처를 중복 없이 모은다."""
        refs: list[str] = []
        for item in state["retrieval"].get("retrieved_docs", []):
            refs.append(item.get("source", "local"))
        for item in state["web_search"].get("web_results", []):
            refs.append(item.get("url") or item.get("source", "web"))
        return self._dedupe_strings(refs)

    @staticmethod
    def _dedupe_records(records: list[dict[str, Any]], unique_key) -> list[dict[str, Any]]:
        """원래 순서를 유지하면서 중복 레코드를 제거한다."""
        seen = set()
        deduped = []
        for record in records:
            key = unique_key(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        """원래 순서를 유지하면서 빈 문자열과 중복 문자열을 제거한다."""
        seen = set()
        deduped = []
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """어휘 기반 검색 점수 계산에 사용할 토큰 집합을 만든다."""
        return set(re.findall(r"[a-zA-Z0-9_\-]+", text.lower()))

    @staticmethod
    def _contains_alias(text: str, aliases: list[str]) -> bool:
        """주어진 텍스트가 alias 후보 중 하나를 포함하는지 반환한다."""
        lowered = text.lower()
        return any(alias in lowered for alias in aliases)

    @staticmethod
    def _technology_aliases(technology: str) -> list[str]:
        """기술명에 대한 검색/매칭용 alias 목록을 반환한다."""
        mapping = {
            "HBM4": ["hbm4", "hbm 4", "hbm"],
            "HBM": ["hbm", "high bandwidth memory"],
            "PIM": ["pim", "processing in memory", "processing-in-memory"],
            "CXL": ["cxl", "compute express link"],
        }
        return mapping.get(technology, [technology.lower()])

    @staticmethod
    def _competitor_aliases(competitor: str) -> list[str]:
        """경쟁사명에 대한 검색/매칭용 alias 목록을 반환한다."""
        mapping = {
            "SK hynix": ["sk hynix", "hynix", "skhynix"],
            "Samsung": ["samsung", "samsung electronics"],
            "Micron": ["micron"],
            "NVIDIA": ["nvidia"],
            "AMD": ["amd", "advanced micro devices"],
            "Intel": ["intel"],
        }
        return mapping.get(competitor, [competitor.lower()])

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """임베딩 벡터 간 cosine similarity를 계산한다."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _is_direct_evidence(text: str) -> bool:
        """문장 조각이 직접 기술 근거처럼 보이는지 분류한다."""
        lowered = text.lower()
        direct_markers = [
            "isscc",
            "hot chips",
            "paper",
            "논문",
            "press release",
            "official",
            "양산",
            "mass production",
            "sample",
            "benchmark",
            "qualification",
            "launch",
            "announce",
            "product",
            "jdec",
            "jedec",
        ]
        return any(marker in lowered for marker in direct_markers)
