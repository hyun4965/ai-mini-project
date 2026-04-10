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
from .models import DecisionOutput, QueryInterpretation
from .services.assessment_service import AssessmentService
from .services.draft_service import DraftService
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
from .supervisor import WorkflowSupervisor


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
- TRL은 NASA가 우주 기술의 개발 단계를 표준화하기 위해 만든 9단계 척도이며, 반도체를 포함한 첨단 산업의 기술 성숙도 판단에 적용한다.
- SWOT는 강하다/약하다처럼 상대적이고 주관적인 판단에 머물 수 있지만, TRL은 "지금 몇 단계인가"라는 절대적 위치를 제공한다.
- TRL 1: 기초 원리 관찰, 아이디어/이론 수준
- TRL 2: 기술 개념 정립, 적용 가능성 검토
- TRL 3: 개념 검증, 실험실 수준 실증
- TRL 4: 부품 검증, 실험실 환경 통합
- TRL 5: 부품 검증(실환경), 유사 환경 통합 테스트
- TRL 6: 시스템 시연, 실제 환경 유사 조건 시연
- TRL 7: 시스템 시제품, 실제 운용 환경 시연
- TRL 8: 시스템 완성, 양산 적합성 검증 완료
- TRL 9: 실제 운용, 상용 양산 및 납품
- TRL 1~3은 논문, 학회 발표, 특허 출원, 개념 검증 자료 등 공개 연구 신호를 중심으로 판단한다.
- TRL 4~6은 공개 정보 기준으로 가장 정보 공백이 큰 구간이며, 수율, 공정 파라미터, 실제 성능 수치가 영업 비밀일 수 있으므로 보수적으로 추정한다.
- 공식 prototype, sample, qualification, pilot, system demonstration 같은 검증 신호가 없으면 TRL 4~6을 높게 주지 않는다.
- TRL 7~9는 고객사 샘플 공급, 양산 발표, 출하, 실적 공시 등 비즈니스 목적의 공개 신호가 있을 때만 제한적으로 검토한다.
- 정확한 기술에 대해 공식 양산/출하/실적 반영 근거가 없는 한 공개 정보만으로 TRL 7 이상을 쉽게 부여하지 않는다.
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
        self.supervisor = WorkflowSupervisor(config, is_listing_heavy=self._is_listing_heavy)
        self.assessment_service = AssessmentService(
            config=config,
            analysis_llm=self.analysis_llm,
            assessment_prompt=ASSESSMENT_PROMPT,
            invoke_llm_with_retry=self._invoke_llm_with_retry,
            logger=self.logger,
        )
        self.draft_service = DraftService(
            config=config,
            draft_llm=self.draft_llm,
            draft_prompt=DRAFT_PROMPT,
            invoke_llm_with_retry=self._invoke_llm_with_retry,
            logger=self.logger,
            collect_references=self._collect_references,
            is_listing_heavy=self._is_listing_heavy,
        )

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
        return self.assessment_service.run(
            state,
            validate_assessment_quality=self._validate_assessment_quality,
            control_update=self._control_update,
        )

    def decision_node(self, state: DecisionInput) -> DecisionUpdate:
        """평가 결과를 바탕으로 R&D 추천을 생성하고 검증한다."""
        self.logger.info("decision.start assessment_rows=%d", len(state["assessment"].get("results", [])))
        decision = self._make_decision(state)
        is_valid, decision_reason, decision_status = self._validate_decision_quality(
            state | {"decision": {**state["decision"], "result": decision}}
        )
        if decision_reason in {
            "missing_decision_rationale",
            "ungrounded_decision_rationale",
            "missing_strategic_actions",
        }:
            self.logger.warning("decision.repair reason=%s", decision_reason)
            decision = self._normalize_decision_output({}, state)
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
        return self.draft_service.run(
            state,
            validate_draft_quality=self._validate_draft_quality,
            control_update=self._control_update,
        )

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
        return self.supervisor.compute_review(state)

    def _validate_information_sufficiency(self, state: SupervisorInput) -> tuple[bool, str]:
        """검색/웹 근거가 최소 커버리지 기준을 만족하는지 확인한다."""
        return self.supervisor.validate_information_sufficiency(state)

    def _validate_analysis_complete(self, state: AssessmentInput | SupervisorInput) -> bool:
        """예상 평가 쌍이 모두 품질 검증을 통과했는지 반환한다."""
        return self.supervisor.validate_analysis_complete(state)

    def _validate_assessment_quality(self, state: AssessmentInput | SupervisorInput) -> tuple[bool, str, str]:
        """평가 커버리지, 근거 품질, 지표 범위를 검증한다."""
        return self.supervisor.validate_assessment_quality(state)

    def _route_analysis_retry(self, state: SupervisorInput, reason: str) -> str:
        """Assessment 품질 검증이 실패했을 때 가장 적절한 재시도 노드를 고른다."""
        return self.supervisor.route_analysis_retry(state, reason)

    def _validate_decision(self, state: DecisionInput | SupervisorInput) -> bool:
        """의사결정 결과가 품질 검증을 통과했는지 반환한다."""
        return self.supervisor.validate_decision(state)

    def _validate_decision_quality(self, state: DecisionInput | SupervisorInput) -> tuple[bool, str, str]:
        """추천 커버리지, 형식, 근거 연결성, 실행안을 검증한다."""
        return self.supervisor.validate_decision_quality(state)

    def _route_decision_retry(self, state: SupervisorInput, reason: str) -> str:
        """Decision 검증이 실패했을 때 재시도할 노드를 고른다."""
        return self.supervisor.route_decision_retry(state, reason)

    def _validate_draft(self, state: SupervisorInput) -> bool:
        """보고서 초안이 품질 검증을 통과했는지 반환한다."""
        return self.supervisor.validate_draft(state)

    def _validate_draft_quality(self, state: DraftInput | SupervisorInput) -> tuple[bool, str, str]:
        """필수 보고서 섹션, 근거 연결성, 분석형 문체를 검증한다."""
        return self.supervisor.validate_draft_quality(state)

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
                    title = self._humanize_path_stem(path.stem)
                elif path.suffix.lower() == ".pdf":
                    reader = PdfReader(str(path))
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    title = self._extract_pdf_title(path, reader, text)
                else:
                    continue
            except (OSError, PdfReadError, ValueError) as exc:
                error = DocumentLoadError(f"failed to load document {path}: {exc}")
                self.logger.warning("document.load.skipped path=%s reason=%s", path, error)
                continue
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": str(path), "title": title}))

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

        if preferred_backend == "huggingface":
            self.logger.warning("embedding.backend.fallback requested=huggingface fallback=hashing")

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
                "title_strategy_version": 2,
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
        haystack = "\n".join(
            [
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("source", ""),
            ]
        ).lower()

        tech_match = any(self._contains_alias(haystack, self._technology_aliases(tech)) for tech in technologies)
        return tech_match

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

    def _collect_references(self, state: DraftInput | SupervisorInput) -> list[str]:
        """보고서에 사용할 로컬/웹 참고 출처를 중복 없이 모은다."""
        refs: list[str] = []
        for item in state["retrieval"].get("retrieved_docs", []):
            title = (item.get("title") or "").strip()
            refs.append(title or self._humanize_path_stem(Path(item.get("source", "local")).stem))
        for item in state["web_search"].get("web_results", []):
            title = (item.get("title") or "").strip()
            source = (item.get("source") or "web").strip()
            source_lower = source.lower()
            url_lower = str(item.get("url") or "").lower()
            if any(
                weak_domain in source_lower or weak_domain in url_lower
                for weak_domain in ("reddit.com", "linkedin.com", "substack.com", "cryptopolitan.com")
            ):
                continue
            if title:
                refs.append(f"{title} ({source})")
            elif item.get("url"):
                refs.append(str(item["url"]))
            else:
                refs.append(source)
        return self._dedupe_strings(refs)

    @staticmethod
    def _humanize_path_stem(stem: str) -> str:
        """파일 stem을 참고문헌용 사람이 읽기 좋은 제목으로 바꾼다."""
        cleaned = re.sub(r"[_\-.]+", " ", stem or "").strip()
        return re.sub(r"\s+", " ", cleaned)

    def _extract_pdf_title(self, path: Path, reader: PdfReader, text: str) -> str:
        """PDF 메타데이터와 첫 페이지 텍스트에서 사람이 읽을 제목을 추정한다."""
        metadata_title = ""
        metadata = getattr(reader, "metadata", None)
        if metadata:
            metadata_title = str(getattr(metadata, "title", "") or "").strip()
        if self._looks_like_reference_title(metadata_title):
            return metadata_title

        lines = [re.sub(r"\s+", " ", line).strip(" -") for line in text.splitlines()[:40]]
        lines = [line for line in lines if line]
        for index, line in enumerate(lines):
            if not self._looks_like_title_candidate(line):
                continue

            combined = line
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                if self._looks_like_title_continuation(next_line):
                    combined = f"{combined.rstrip('- ')} {next_line.lstrip('- ')}"
            if self._looks_like_reference_title(combined):
                return combined

        return self._humanize_path_stem(path.stem)

    @staticmethod
    def _looks_like_title_candidate(line: str) -> bool:
        """논문 제목 시작 줄처럼 보이는지 판별한다."""
        lowered = line.lower()
        if len(line) < 12:
            return False
        if any(token in lowered for token in ("journal", "issn", "vol ", "abstract", "keywords", "doi", "www.", "http")):
            return False
        if re.fullmatch(r"\d+", line):
            return False
        if line.count(",") >= 3:
            return False
        if not re.search(r"[A-Za-z가-힣]", line):
            return False
        return True

    @staticmethod
    def _looks_like_title_continuation(line: str) -> bool:
        """논문 제목의 다음 줄처럼 보이는지 판별한다."""
        lowered = line.lower()
        if len(line) < 3:
            return False
        if any(token in lowered for token in ("abstract", "keywords", "issn", "vol ", "journal", "university", "research group", "inc.", "corporation", "usa")):
            return False
        if line.count(",") >= 2:
            return False
        return bool(re.search(r"[A-Za-z가-힣]", line))

    @staticmethod
    def _looks_like_reference_title(text: str) -> bool:
        """레퍼런스 제목으로 쓸 만한 문장인지 판별한다."""
        normalized = re.sub(r"\s+", " ", (text or "")).strip(" .-")
        if len(normalized) < 12:
            return False
        if re.fullmatch(r"[A-Z]{2,}-?\d{2,}.*", normalized):
            return False
        if normalized.lower().startswith(("issn", "journal", "vol ", "abstract")):
            return False
        return True

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
            "conference",
            "press release",
            "양산",
            "mass production",
            "sample",
            "qualification",
            "customer validation",
            "prototype",
            "demonstration",
            "test chip",
            "shipment",
            "shipping",
            "출하",
            "시제품",
            "시연",
            "jdec",
            "jedec",
        ]
        return any(marker in lowered for marker in direct_markers)
