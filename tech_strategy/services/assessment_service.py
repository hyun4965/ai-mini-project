from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import StrategyConfig
from ..errors import LLMServiceError
from ..models import AssessmentResult
from ..state_contracts import AssessmentInput, AssessmentUpdate


class AssessmentService:
    """Evidence synthesis와 TRL/Threat 평가를 담당하는 도메인 서비스."""

    def __init__(
        self,
        *,
        config: StrategyConfig,
        analysis_llm,
        assessment_prompt: str,
        invoke_llm_with_retry,
        logger,
    ) -> None:
        self.config = config
        self.analysis_llm = analysis_llm
        self.assessment_prompt = assessment_prompt
        self._invoke_llm_with_retry = invoke_llm_with_retry
        self.logger = logger

    def run(
        self,
        state: AssessmentInput,
        *,
        validate_assessment_quality,
        control_update,
    ) -> AssessmentUpdate:
        """assessment 노드 전체 실행을 담당한다."""
        self.logger.info(
            "assessment.start technologies=%d competitors=%d",
            len(state["scope"]["target_technologies"] or [state["scope"]["target_technology"]]),
            len(state["scope"]["target_competitors"]),
        )
        bundle = self._build_evidence_bundle(state)
        assessment_results: list[dict[str, Any]] = []

        scope = state["scope"]
        technologies = scope["target_technologies"] or [scope["target_technology"]]
        competitors = scope["target_competitors"]
        for technology in technologies:
            for competitor in competitors:
                evidence = bundle.get(technology, {}).get(
                    competitor,
                    {"direct_evidence": [], "indirect_evidence": [], "sources": []},
                )
                assessment_results.append(
                    self._assess_pair(
                        user_query=state["user_query"],
                        technology=technology,
                        competitor=competitor,
                        evidence=evidence,
                    )
                )

        analysis_complete, analysis_reason, coverage_status = validate_assessment_quality(
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
            "control": control_update("assessment", coverage_status=coverage_status),
            "analysis_log": [
                f"[assessment] assessed_pairs={len(assessment_results)} "
                f"analysis_complete={analysis_complete} reason={analysis_reason or 'ok'}"
            ],
        }

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

                    item_source_type = item.get("source_type", "")
                    competitor_match = self._contains_alias(haystack, self._competitor_aliases(competitor))
                    if item_source_type != "retrieval" and not competitor_match:
                        continue

                    entry = f'{item.get("title", "Untitled")}: {item.get("content", "")[:500]}'
                    if item_source_type == "retrieval" and not competitor_match:
                        indirect_evidence.append(f"[기술 베이스라인] {entry}")
                        if item.get("url"):
                            sources.append(item["url"])
                        else:
                            sources.append(item.get("source", "unknown"))
                        continue

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
        payload_text = self._format_evidence_for_prompt(
            user_query=user_query,
            technology=technology,
            competitor=competitor,
            evidence=evidence,
        )

        try:
            result = self._invoke_llm_with_retry(
                f"assessment:{technology}:{competitor}",
                lambda: schema.invoke(
                    [
                        SystemMessage(content=self.assessment_prompt),
                        HumanMessage(content=payload_text),
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

    def _format_evidence_for_prompt(
        self,
        *,
        user_query: str,
        technology: str,
        competitor: str,
        evidence: dict[str, Any],
    ) -> str:
        """LLM assessment 입력용 evidence 텍스트를 정제해 구성한다."""
        direct = self._prepare_prompt_items(evidence.get("direct_evidence", []), limit=4, max_chars=320)
        indirect = self._prepare_prompt_items(evidence.get("indirect_evidence", []), limit=5, max_chars=240)
        sources = self._prepare_prompt_items(evidence.get("sources", []), limit=6, max_chars=160)

        direct_block = "\n".join(f"- {item}" for item in direct) if direct else "- 없음"
        indirect_block = "\n".join(f"- {item}" for item in indirect) if indirect else "- 없음"
        source_block = "\n".join(f"- {item}" for item in sources) if sources else "- 없음"

        return (
            f"사용자 질의: {self._clean_text_for_llm(user_query, max_chars=400)}\n"
            f"분석 기술: {technology}\n"
            f"분석 경쟁사: {competitor}\n\n"
            "직접 근거:\n"
            f"{direct_block}\n\n"
            "간접 지표:\n"
            f"{indirect_block}\n\n"
            "출처:\n"
            f"{source_block}\n\n"
            "위 근거만 사용해 TRL, Threat, uncertainty를 구조화해 평가하라."
        )

    def _prepare_prompt_items(self, values: list[str], *, limit: int, max_chars: int) -> list[str]:
        """프롬프트에 넣기 전 문자열 목록을 정제·절단한다."""
        cleaned_items: list[str] = []
        for value in self._dedupe_strings(values):
            cleaned = self._clean_text_for_llm(value, max_chars=max_chars)
            if cleaned:
                cleaned_items.append(cleaned)
            if len(cleaned_items) >= limit:
                break
        return cleaned_items

    @staticmethod
    def _clean_text_for_llm(value: Any, *, max_chars: int) -> str:
        """LLM 요청 안정성을 위해 제어문자와 과도한 길이를 제거한다."""
        if not isinstance(value, str):
            return ""
        cleaned = value.replace("\x00", " ")
        cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > max_chars:
            cleaned = f"{cleaned[: max_chars - 3].rstrip()}..."
        return cleaned

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
        if direct_count == 0:
            trl_ceiling = min(trl_ceiling, 3)
            if "직접 근거" not in ceiling_reason:
                ceiling_reason = (
                    "직접 근거가 없어 공개 자료 기준 TRL 3을 상한으로 둔다. "
                    f"{ceiling_reason}".strip()
                )
        if trl_level > trl_ceiling:
            trl_level = trl_ceiling

        commercialization_signal = raw.get("commercialization_signal") or self._commercialization_signal_from_trl(trl_level)
        if commercialization_signal not in {"Research", "Prototype", "Pilot", "Production", "Unclear"}:
            commercialization_signal = self._commercialization_signal_from_trl(trl_level)
        commercialization_signal = self._commercialization_signal_from_trl(trl_level)
        if direct_count == 0:
            commercialization_signal = "Research"

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
        if direct_count == 0:
            trl_confidence = min(trl_confidence, 0.45)
        if 4 <= trl_level <= 6:
            trl_confidence = min(trl_confidence, 0.68)
        elif trl_level >= 7:
            trl_confidence = min(trl_confidence, 0.62)

        relative_position = raw.get("relative_position_to_sk_hynix")
        if competitor == "SK hynix":
            relative_position = "Comparable"
        elif direct_count == 0:
            relative_position = "Unclear"
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
        high_trl_uncertainty_note = (
            "TRL 7 이상은 고객사 샘플 공급, 양산 발표, 출하, 실적 공시 같은 공개 비즈니스 신호를 바탕으로 제한적으로 추정한 값이다. "
            "다만 수율, 원가, 공정 안정성, 실제 고객 검증 범위는 비공개일 수 있어 보수적으로 해석해야 한다."
        )
        uncertainty_note = self._prefer_korean_text(raw.get("uncertainty_note"), "")
        if direct_count == 0 and not uncertainty_note:
            uncertainty_note = (
                "직접 근거가 확인되지 않아 논문, 특허, 학회 발표, 채용·투자·표준화 활동 같은 간접 지표 중심으로 보수적으로 추정했다. "
                "따라서 본 평가는 저신뢰 참고치로 해석해야 한다."
            )
        if 4 <= trl_level <= 6 and not uncertainty_note:
            uncertainty_note = default_uncertainty_note
        if trl_level >= 7 and not uncertainty_note:
            uncertainty_note = high_trl_uncertainty_note
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
        research_markers = (
            "paper",
            "논문",
            "isscc",
            "hot chips",
            "patent",
            "특허",
            "proof of concept",
            "concept verification",
            "개념 검증",
        )
        prototype_markers = (
            "prototype",
            "demonstration",
            "test chip",
            "lab validation",
            "integration test",
            "benchmark",
            "시제품",
            "시연",
            "실험실",
            "통합 테스트",
        )
        sampling_markers = (
            "sample",
            "sampling",
            "qualification",
            "customer validation",
            "pilot",
            "system demo",
            "field trial",
            "prototype",
            "샘플",
            "검증",
            "고객 검증",
            "파일럿",
        )
        production_markers = (
            "mass production",
            "volume production",
            "commercial production",
            "shipping",
            "customer shipment",
            "delivered",
            "revenue",
            "earnings",
            "sales",
            "product launch",
            "launched",
            "양산",
            "출하",
            "납품",
            "상용",
            "출시",
            "매출",
            "실적",
            "공시",
        )
        has_official_signal = any(marker in evidence_text for marker in official_markers)
        has_research_signal = any(marker in evidence_text for marker in research_markers)
        has_prototype_signal = any(marker in evidence_text for marker in prototype_markers)
        has_sampling_signal = any(marker in evidence_text for marker in sampling_markers)
        has_production_signal = any(marker in evidence_text for marker in production_markers)
        unique_sources = {source.strip().lower() for source in sources if source and source.strip()}
        source_count = len(unique_sources)

        if has_official_signal and has_production_signal and source_count >= 2:
            return 7, "고객사 샘플 공급, 양산 발표, 출하, 실적 공시 같은 공개 비즈니스 신호가 교차 확인되어 TRL 7까지는 제한적으로 추정할 수 있으나, 내부 수율·공정·원가 자료 부재로 TRL 8~9는 보수적으로 보류한다."
        if has_official_signal and has_sampling_signal and source_count >= 2:
            return 6, "공개 공식 자료에서 샘플·qualification·pilot·system demonstration 신호가 교차 확인되어 TRL 6까지 추정한다."
        if has_official_signal and (has_prototype_signal or has_sampling_signal):
            return 5, "공개 자료에서 prototype 또는 실환경 유사 검증 신호가 확인되지만, 실제 운용 환경 검증과 고객 채택까지는 확인되지 않아 TRL 5를 상한으로 둔다."
        if has_prototype_signal:
            return 4, "실험실 수준의 부품 검증 또는 통합 시연 신호가 확인되어 공개 자료 기준 TRL 4를 상한으로 둔다."
        if has_research_signal or indirect_evidence:
            return 3, "논문, 학회, 특허, 간접 지표 중심의 공개 연구 신호를 근거로 TRL 3을 상한으로 둔다."
        if direct_evidence:
            return 3, "직접 언급은 있으나 실험실 검증 신호가 부족해 공개 자료 기준 TRL 3을 상한으로 둔다."
        return 2, "개념 정립 수준의 제한적 공개 정보만 있어 TRL 2를 상한으로 둔다."

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
    def _contains_alias(text: str, aliases: list[str]) -> bool:
        """주어진 텍스트가 alias 후보 중 하나를 포함하는지 반환한다."""
        lowered = text.lower()
        return any(alias in lowered for alias in aliases)

    @staticmethod
    def _technology_aliases(technology: str) -> list[str]:
        """기술명에 대한 웹 검색/필터링용 alias 목록을 반환한다."""
        mapping = {
            "HBM4": ["hbm4", "hbm 4", "high bandwidth memory"],
            "HBM": ["hbm", "high bandwidth memory"],
            "PIM": ["processing in memory", "processing-in-memory", "pim memory", "dram pim", "hbm-pim"],
            "CXL": ["cxl", "compute express link"],
        }
        return mapping.get(technology, [technology.lower()])

    @staticmethod
    def _competitor_aliases(competitor: str) -> list[str]:
        """경쟁사명에 대한 alias 목록을 반환한다."""
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
