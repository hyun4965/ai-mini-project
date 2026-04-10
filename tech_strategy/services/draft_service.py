from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import StrategyConfig
from ..errors import LLMServiceError
from ..report_structure import REQUIRED_REPORT_HEADINGS
from ..state_contracts import DraftInput, DraftUpdate, SupervisorInput


class DraftService:
    """보고서 초안 생성과 fallback 조립을 담당하는 서비스."""

    def __init__(
        self,
        *,
        config: StrategyConfig,
        draft_llm,
        draft_prompt: str,
        invoke_llm_with_retry,
        logger,
        collect_references,
        is_listing_heavy,
    ) -> None:
        self.config = config
        self.draft_llm = draft_llm
        self.draft_prompt = draft_prompt
        self._invoke_llm_with_retry = invoke_llm_with_retry
        self.logger = logger
        self._collect_references = collect_references
        self._is_listing_heavy = is_listing_heavy

    def run(
        self,
        state: DraftInput,
        *,
        validate_draft_quality,
        control_update,
    ) -> DraftUpdate:
        """draft 노드 전체 실행을 담당한다."""
        draft_version = state["draft"]["version"] + 1
        self.logger.info("draft.start version=%d", draft_version)
        markdown = self._draft_report(state)
        quality = self._score_draft(markdown)
        is_valid, draft_reason, draft_status = validate_draft_quality(
            state | {"draft": {**state["draft"], "markdown_text": markdown}}
        )
        if not is_valid:
            fallback_markdown = self._build_fallback_draft(state)
            fallback_quality = self._score_draft(fallback_markdown)
            fallback_valid, fallback_reason, fallback_status = validate_draft_quality(
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
            "control": control_update("draft"),
            "analysis_log": [
                f"[draft] version={draft_version} quality={quality:.2f} "
                f"draft_valid={is_valid} reason={draft_reason or 'ok'} status={draft_status}"
            ],
        }

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
                        SystemMessage(content=self.draft_prompt),
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
            if not all(section in text for section in REQUIRED_REPORT_HEADINGS):
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
            aliases = {
                "HBM": {"HBM", "HBM4"},
                "PIM": {"PIM"},
                "CXL": {"CXL"},
            }
            return [row for row in assessments if row.get("technology") in aliases[report_technology]]

        def build_tech_section(report_technology: str) -> str:
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
            status_lines = " ".join(self._soften_status_summary_for_report(row) for row in rows[:3])
            return (
                f"{overview} 공개 근거 기준 평균 TRL은 {avg_trl:.1f} 수준으로 정리되며, 직접 근거 {direct_total}건과 간접 지표 {indirect_total}건이 연결되었다. "
                f"주요 비교 대상은 {top_competitors}이며, 현재 개발 방향은 {direction or '불확실'} 신호로 요약된다. "
                f"{status_lines}"
            )

        competitor_direction_lines = [
            f"- {row['competitor']} / {row['technology']}: {row.get('competitor_level_summary', '')} "
            f"직접 근거 {row.get('direct_evidence_count', 0)}건, 간접 지표 {row.get('indirect_evidence_count', 0)}건."
            for row in assessments
        ]
        trl_lines = [
            f"- {row['technology']} / {row['competitor']}: TRL {row['trl_level']} "
            f"(신뢰도 {row.get('trl_confidence', 0.0):.2f}), 상대 위치 {self._label_relative_position(row.get('relative_position_to_sk_hynix', 'Unclear'))}."
            for row in assessments
        ]
        threat_lines = [
            f"- {row['technology']} / {row['competitor']}: 위협 수준 {self._label_threat_level(row['threat_level'])} "
            f"(점수 {row['threat_score']:.2f}), 시장 영향력 {row['market_impact']:.2f}, 경쟁 강도 {row['competition_intensity']:.2f}. "
            f"{row.get('threat_rationale', '')}"
            for row in assessments
        ]
        decision_lines = [
            f"- {rec['technology']}: 권고 {rec['rd_feasibility']} / 우선순위 {rec['priority_level']} / "
            f"판단 근거: {rec['decision_rationale']}"
            for rec in decision.get("recommendations", [])
        ]

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

TRL은 NASA가 우주 기술의 개발 단계를 표준화하기 위해 만든 9단계 기술 성숙도 척도를 반도체 기술 분석에 적용한 기준이다. 본 보고서는 SWOT처럼 강하다/약하다에 가까운 상대적 인상에 머무르지 않고, 각 기술이 지금 몇 단계에 있는지라는 절대적 위치를 제시하기 위해 TRL을 채택했다.

- TRL 1: 기초 원리 관찰, 아이디어/이론 수준
- TRL 2: 기술 개념 정립, 적용 가능성 검토
- TRL 3: 개념 검증, 실험실 수준 실증
- TRL 4: 부품 검증, 실험실 환경 통합
- TRL 5: 부품 검증(실환경), 유사 환경 통합 테스트
- TRL 6: 시스템 시연, 실제 환경 유사 조건 시연
- TRL 7: 시스템 시제품, 실제 운용 환경 시연
- TRL 8: 시스템 완성, 양산 적합성 검증 완료
- TRL 9: 실제 운용, 상용 양산 및 납품

공개 정보 관점에서 TRL 1~3은 논문, 학회, 특허를 통해 비교적 관찰 가능하지만, TRL 4~6은 수율·공정·실성능 데이터가 비공개인 경우가 많아 가장 큰 추정 구간이다. TRL 7~9는 고객사 샘플 공급, 양산 발표, 실적 공시 등 일부 비즈니스 신호를 통해 제한적으로만 확인 가능하므로, 본 보고서는 공개 근거로 확인 가능한 상한을 보수적으로 적용한다.

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

기술별 TRL 수준 비교는 동일한 1~9 기준을 적용해 경쟁사 간 상대 위치를 맞춰 보는 데 목적이 있다. 다만 공개 정보 관점에서 TRL 1~3은 연구 신호로 비교적 관찰 가능하지만, TRL 4~6은 발표나 시연만으로 내부 검증 완료 여부를 확정하기 어렵기 때문에 가장 큰 추정 구간에 해당한다. 또한 TRL 7~9는 고객 샘플, 양산, 출하, 실적 공시 등 비즈니스 목적의 공개 신호가 있을 때만 제한적으로 검토했다.

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
- TRL 4~6은 수율, 공정 파라미터, 실제 성능 수치가 핵심 영업 비밀일 수 있어 가장 큰 정보 공백 구간에 해당함
- TRL 7~9는 고객사 샘플 공급, 양산 발표, 실적 공시 등 일부 공개 신호로 제한적으로 확인 가능하지만, 내부 검증 데이터 없이 정밀 판정하기는 어려움
- 수율, 공정, 성능 데이터는 비공개 영역이어서 정확한 평가에 한계 존재
- 특허 출원 패턴, 학회 발표 빈도 변화, 채용 공고 키워드, 투자 및 협력 발표 같은 간접 지표를 함께 사용했음

## REFERENCE

{reference_lines}
"""

    def _soften_status_summary_for_report(self, row: dict[str, Any]) -> str:
        """보고서 서술이 보수적 TRL 판단과 충돌하지 않도록 status summary를 완화한다."""
        summary = (row.get("current_status_summary") or "").strip()
        if not summary:
            return ""

        trl_level = int(row.get("trl_level") or 0)
        commercialization_markers = ("양산", "출하", "샘플", "production", "shipping", "shipment", "sample")
        if trl_level <= 3 and any(marker in summary.lower() for marker in commercialization_markers):
            return (
                f"{summary} 다만 본 보고서는 공개 발표만으로 내부 검증, 고객 qualification, 지속 출하 여부를 확정하기 어렵다고 보아 "
                f"TRL {trl_level}의 보수적 해석을 유지했다."
            )
        return summary

    def _score_draft(self, markdown: str) -> float:
        """필수 표시문자를 기준으로 단순 보고서 품질 점수를 계산한다."""
        criteria = 0.0
        criteria += sum(1 for section in REQUIRED_REPORT_HEADINGS if section in markdown) / len(REQUIRED_REPORT_HEADINGS) * 0.5
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
    def _has_excessive_english_narrative(cls, markdown: str) -> bool:
        """보고서 본문에서 영어 서술 비중이 과도하면 fallback 대상으로 간주한다."""
        body = markdown.split("## REFERENCE", maxsplit=1)[0]
        hangul_count = len(re.findall(r"[가-힣]", body))
        english_count = len(re.findall(r"[A-Za-z]", body))
        return hangul_count == 0 or english_count > hangul_count

    @staticmethod
    def _label_commercialization_signal(signal: str) -> str:
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
        mapping = {
            "Ahead": "우위",
            "Comparable": "유사",
            "Behind": "열위",
            "Unclear": "불확실",
        }
        return mapping.get(position, position or "불확실")

    @staticmethod
    def _label_threat_level(level: str) -> str:
        mapping = {
            "High": "High(높음)",
            "Medium": "Medium(중간)",
            "Low": "Low(낮음)",
        }
        return mapping.get(level, level or "Unclear")
