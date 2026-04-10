from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import StrategyConfig
from .report_structure import REQUIRED_REPORT_HEADINGS


class WorkflowSupervisor:
    """Supervisor 정책, 품질 검증, 재시도 라우팅을 담당한다."""

    def __init__(self, config: StrategyConfig, *, is_listing_heavy) -> None:
        self.config = config
        self._is_listing_heavy = is_listing_heavy

    def compute_review(self, state: dict[str, Any]) -> dict[str, Any]:
        """워크플로우 검증 기준을 평가하고 다음 단계를 계산한다."""
        info_ok, coverage_status = self.validate_information_sufficiency(state)
        analysis_ok, analysis_reason, analysis_status = self.validate_assessment_quality(state)
        decision_ok, decision_reason, decision_status = self.validate_decision_quality(state)
        draft_ok, draft_reason, draft_status = self.validate_draft_quality(state)

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
            next_step = (
                "web_search"
                if web_search.get("source_diversity", 0) < self.config.min_source_diversity
                or not web_search.get("has_counter_evidence")
                else "retrieval"
            )
            retry_count += 1
        elif info_ok and not state["assessment"].get("results"):
            next_step = "assessment"
        elif not analysis_ok:
            current_stage = control.get("workflow_stage")
            if current_stage == "assessment":
                retry_count += 1
                next_step = self.route_analysis_retry(state, analysis_reason)
            elif current_stage in {"retrieval", "web_search"}:
                # 검색 보강 이후에는 같은 검색 노드를 반복하지 말고
                # 최신 근거로 assessment를 다시 실행해 분석을 갱신한다.
                next_step = "assessment"
            else:
                next_step = self.route_analysis_retry(state, analysis_reason)
        elif not decision_ok:
            if control.get("workflow_stage") == "decision":
                retry_count += 1
            next_step = self.route_decision_retry(state, decision_reason)
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

        if retry_count >= control["max_iteration"] and (
            not info_ok or not analysis_ok or not decision_ok or not draft_ok or output.get("format_error")
        ):
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

    def validate_information_sufficiency(self, state: dict[str, Any]) -> tuple[bool, str]:
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

    def validate_analysis_complete(self, state: dict[str, Any]) -> bool:
        """예상 평가 쌍이 모두 품질 검증을 통과했는지 반환한다."""
        analysis_ok, _, _ = self.validate_assessment_quality(state)
        return analysis_ok

    def validate_assessment_quality(self, state: dict[str, Any]) -> tuple[bool, str, str]:
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

        result_index = {(item.get("technology"), item.get("competitor")): item for item in results}

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
                    return False, "insufficient_evidence", (
                        f"{technology}/{competitor} evidence total={total_evidence} sources={len(sources)}"
                    )

                trl_level = item.get("trl_level")
                if not isinstance(trl_level, int) or not 1 <= trl_level <= 9:
                    return False, "invalid_trl_range", f"{technology}/{competitor} trl={trl_level}"
                if not item.get("trl_rationale"):
                    return False, "missing_trl_rationale", f"{technology}/{competitor} missing TRL rationale"
                if not direct:
                    if trl_level > 3:
                        return False, "invalid_trl_support", (
                            f"{technology}/{competitor} cannot exceed TRL 3 without direct evidence"
                        )
                    uncertainty_note = (item.get("uncertainty_note") or "").strip()
                    if not uncertainty_note:
                        return False, "missing_trl_uncertainty", (
                            f"{technology}/{competitor} missing uncertainty for indirect-only assessment"
                        )
                trl_supported, trl_support_reason = self._validate_trl_support(
                    trl_level=trl_level,
                    direct=direct,
                    indirect=indirect,
                    sources=sources,
                )
                if not trl_supported:
                    return False, "invalid_trl_support", f"{technology}/{competitor} {trl_support_reason}"
                if 4 <= trl_level <= 6 and not item.get("uncertainty_note"):
                    return False, "missing_trl_uncertainty", f"{technology}/{competitor} missing TRL 4-6 uncertainty"

                if not item.get("threat_rationale"):
                    return False, "missing_threat_rationale", f"{technology}/{competitor} missing threat rationale"

                for metric_name in ("threat_score", "market_impact", "competition_intensity"):
                    metric_value = item.get(metric_name)
                    if not isinstance(metric_value, (int, float)) or not 0.0 <= float(metric_value) <= 1.0:
                        return False, "non_comparable_threat", (
                            f"{technology}/{competitor} invalid {metric_name}={metric_value}"
                        )

        return True, "", f"assessment pairs validated {expected}/{expected}"

    def route_analysis_retry(self, state: dict[str, Any], reason: str) -> str:
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
        if reason in {"missing_trl_uncertainty", "invalid_trl_range", "missing_trl_rationale", "invalid_trl_support"}:
            if (
                web_search.get("source_diversity", 0) < self.config.min_source_diversity
                or web_search.get("freshness_score", 0.0) < self.config.min_recent_ratio
            ):
                return "web_search"
            return "retrieval"
        if reason in {"missing_threat_rationale", "non_comparable_threat"}:
            return "web_search" if not web_search.get("has_counter_evidence") else "retrieval"
        return "assessment"

    def _validate_trl_support(
        self,
        *,
        trl_level: int,
        direct: list[str],
        indirect: list[str],
        sources: list[str],
    ) -> tuple[bool, str]:
        """TRL 단계가 공개 근거 수준과 맞는지 단계별로 검증한다."""
        evidence_text = " ".join([*direct, *indirect, *sources]).lower()
        unique_sources = {source.strip().lower() for source in sources if source and source.strip()}
        source_count = len(unique_sources)

        official_markers = (
            "samsung.com",
            "skhynix.com",
            "micron.com",
            "jedec.org",
            "cxlconsortium.org",
            "computeexpresslink.org",
            "press release",
            "보도자료",
            "공식",
        )
        prototype_markers = (
            "prototype",
            "demonstration",
            "test chip",
            "benchmark",
            "시제품",
            "시연",
            "통합 테스트",
        )
        pilot_markers = (
            "sample",
            "sampling",
            "qualification",
            "customer validation",
            "pilot",
            "system demo",
            "field trial",
            "샘플",
            "고객 검증",
            "파일럿",
        )
        business_markers = (
            "mass production",
            "volume production",
            "commercial production",
            "shipping",
            "customer shipment",
            "revenue",
            "earnings",
            "sales",
            "양산",
            "출하",
            "납품",
            "매출",
            "실적",
            "공시",
        )
        research_markers = (
            "paper",
            "논문",
            "isscc",
            "hot chips",
            "patent",
            "특허",
            "proof of concept",
            "개념 검증",
        )

        has_official = any(marker in evidence_text for marker in official_markers)
        has_prototype = any(marker in evidence_text for marker in prototype_markers)
        has_pilot = any(marker in evidence_text for marker in pilot_markers)
        has_business = any(marker in evidence_text for marker in business_markers)
        has_research = any(marker in evidence_text for marker in research_markers)

        if trl_level <= 3:
            if direct or indirect or has_research:
                return True, "research-stage support is present"
            return False, "trl 1-3 requires at least minimal public research signal"
        if trl_level == 4:
            if has_prototype:
                return True, "trl 4 lab-level validation signal present"
            return False, "trl 4 requires prototype, test-chip, demonstration, or similar lab validation signal"
        if trl_level == 5:
            if has_official and (has_prototype or has_pilot):
                return True, "trl 5 prototype or similar-environment validation signal present"
            return False, "trl 5 requires official prototype or similar-environment validation signal"
        if trl_level == 6:
            if has_official and has_pilot and source_count >= 2:
                return True, "trl 6 pilot or qualification signal cross-validated"
            return False, "trl 6 requires official pilot/qualification signal and at least two sources"
        if trl_level >= 7:
            if has_official and has_business and source_count >= 2:
                return True, "trl 7+ business-stage public signal cross-validated"
            return False, "trl 7+ requires public business signal such as sample supply, production, shipment, or earnings disclosure with multi-source support"
        return False, "unsupported trl level"

    def validate_decision(self, state: dict[str, Any]) -> bool:
        """의사결정 결과가 품질 검증을 통과했는지 반환한다."""
        decision_ok, _, _ = self.validate_decision_quality(state)
        return decision_ok

    def validate_decision_quality(self, state: dict[str, Any]) -> tuple[bool, str, str]:
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

        recommendation_index = {item.get("technology"): item for item in recommendations if item.get("technology")}

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
            grounded_terms = ["trl", "threat", "evidence", "competitor", "위협", "근거", "경쟁", "성숙", "trl"]
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

    def route_decision_retry(self, state: dict[str, Any], reason: str) -> str:
        """Decision 검증이 실패했을 때 재시도할 노드를 고른다."""
        if reason in {"insufficient_assessment", "missing_competitor_link"}:
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

    def validate_draft(self, state: dict[str, Any]) -> bool:
        """보고서 초안이 품질 검증을 통과했는지 반환한다."""
        draft_ok, _, _ = self.validate_draft_quality(state)
        return draft_ok

    def validate_draft_quality(self, state: dict[str, Any]) -> tuple[bool, str, str]:
        """필수 보고서 섹션, 근거 연결성, 분석형 문체를 검증한다."""
        markdown = state["draft"].get("markdown_text", "")
        if not markdown:
            return False, "missing_draft", "draft report is empty"

        if not all(section in markdown for section in REQUIRED_REPORT_HEADINGS):
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
