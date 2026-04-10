from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from typing import Any

from langchain_tavily import TavilySearch

from ..config import StrategyConfig
from ..state_contracts import WebSearchInput, WebSearchUpdate


class WebSearchService:
    """Tavily 검색, 결과 정규화, 웹 검색 품질 점수 계산을 담당한다."""

    def __init__(self, config: StrategyConfig) -> None:
        """워크플로우 설정을 바탕으로 선택적 Tavily 검색 도구를 초기화한다."""
        self.config = config
        if not os.environ.get("TAVILY_API_KEY"):
            self.search_tool = None
            return
        try:
            self.search_tool = TavilySearch(
                max_results=config.tavily_max_results,
                search_depth=config.tavily_search_depth,
            )
        except Exception:
            self.search_tool = None

    @staticmethod
    def _control_update(
        stage: str,
        *,
        control: dict[str, Any] | None = None,
        rewrite_history: list[str] | None = None,
    ) -> dict[str, Any]:
        """질의 재작성 이력을 유지하면서 제어 상태 업데이트를 만든다."""
        update = {"workflow_stage": stage}
        if rewrite_history:
            update["query_rewrite_history"] = [
                *(control or {}).get("query_rewrite_history", []),
                *rewrite_history,
            ]
        return update

    def run(self, state: WebSearchInput) -> WebSearchUpdate:
        """웹 검색 질의를 실행하고 중첩 LangGraph 상태 업데이트를 반환한다."""
        current_plan = dict(state["query_plan"])
        base_queries, counter_queries = self._build_balanced_web_queries(current_plan, state)
        all_queries = base_queries + counter_queries
        search_errors: list[str] = []

        if self.search_tool is None:
            attempt = state["web_search"].get("attempt", 0) + 1
            return {
                "query_plan": current_plan,
                "web_search": {
                    "queries": all_queries,
                    "web_results": [],
                    "source_diversity": 0,
                    "freshness_score": 0.0,
                    "source_reliability_score": 0.0,
                    "has_counter_evidence": False,
                    "bias_risk_score": 1.0,
                    "competitor_coverage": {},
                    "balanced_company_coverage": False,
                    "is_success": False,
                    "failure_reason": "web_search_unavailable",
                    "attempt": attempt,
                },
                "control": self._control_update("web_search", control=state["control"]),
                "analysis_log": [
                    "[web_search] unavailable: TAVILY_API_KEY is not configured or the search tool could not be initialized"
                ],
            }

        results: list[dict[str, Any]] = []
        for query in all_queries:
            stance = "counter" if query in counter_queries else "supportive"
            for item in self._search_web(query, stance, search_errors):
                results.append(item)

        results = self._dedupe_records(
            results,
            unique_key=lambda item: item.get("url") or f'{item["source"]}:{item["title"]}',
        )
        results = [item for item in results if self._matches_scope(item, state)]
        results = self._prioritize_results(results)

        domains = [
            self._extract_domain(item.get("url", "")) or item["source"].lower()
            for item in results
        ]
        domain_counts = Counter(domains)
        source_diversity = len(domain_counts)
        freshness_score = self._compute_freshness_score(results)
        has_counter_evidence = any(item.get("stance") == "counter" for item in results)
        source_reliability_score = (
            round(sum(item.get("source_reliability_score", 0.0) for item in results) / len(results), 4)
            if results
            else 0.0
        )
        competitors = state["scope"].get("target_competitors", [])
        competitor_coverage = self._compute_competitor_coverage(results, competitors)
        balanced_company_coverage = self._is_company_coverage_balanced(
            competitor_coverage,
            competitors,
        )
        domain_concentration = (max(domain_counts.values()) / len(results)) if results else 1.0
        competitor_concentration = self._compute_competitor_concentration(competitor_coverage)
        bias_risk_score = round(max(domain_concentration, competitor_concentration), 4)
        is_search_success = (
            len(results) >= self.config.min_web_results
            and source_diversity >= self.config.min_source_diversity
            and freshness_score >= self.config.min_recent_ratio
            and has_counter_evidence
            and source_reliability_score >= self.config.min_source_reliability_score
            and bias_risk_score <= self.config.max_bias_risk_score
            and balanced_company_coverage
        )

        failure_reason = ""
        rewrite_history: list[str] = []
        attempt = state["web_search"].get("attempt", 0)
        if not is_search_success:
            failure_reason = self._detect_web_search_failure_reason(
                source_diversity=source_diversity,
                freshness_score=freshness_score,
                has_counter_evidence=has_counter_evidence,
                source_reliability_score=source_reliability_score,
                bias_risk_score=bias_risk_score,
                balanced_company_coverage=balanced_company_coverage,
            )
            if not results and search_errors:
                failure_reason = "web_search_api_error"
            current_plan["web_queries"] = self._rewrite_web_queries(
                interpretation=current_plan,
                reason=failure_reason,
            )
            current_plan["counter_queries"] = self._rewrite_counter_queries(
                interpretation=current_plan,
                reason=failure_reason,
            )
            attempt += 1
            rewrite_history = [
                f"[rewrite][web] reason={failure_reason} web_queries={current_plan['web_queries']} counter_queries={current_plan['counter_queries']}"
            ]

        return {
            "query_plan": current_plan,
            "web_search": {
                "queries": all_queries,
                "web_results": results,
                "source_diversity": source_diversity,
                "freshness_score": freshness_score,
                "source_reliability_score": source_reliability_score,
                "has_counter_evidence": has_counter_evidence,
                "bias_risk_score": bias_risk_score,
                "competitor_coverage": competitor_coverage,
                "balanced_company_coverage": balanced_company_coverage,
                "is_success": is_search_success,
                "failure_reason": failure_reason,
                "attempt": attempt,
            },
            "control": self._control_update(
                "web_search",
                control=state["control"],
                rewrite_history=rewrite_history,
            ),
            "analysis_log": [
                f"[web_search] queries={len(all_queries)} results={len(results)} sources={source_diversity} "
                f"recent={freshness_score:.2f} reliability={source_reliability_score:.2f} "
                f"bias={bias_risk_score:.2f} counter={has_counter_evidence} balanced={balanced_company_coverage} "
                f"errors={len(search_errors)} reason={failure_reason or 'ok'}"
            ],
        }

    def _build_balanced_web_queries(
        self,
        interpretation: dict[str, Any],
        state: WebSearchInput,
    ) -> tuple[list[str], list[str]]:
        """계획된 질의에 경쟁사별 긍정/반증 질의를 합쳐 균형 있게 만든다."""
        planned_web = list(interpretation.get("web_queries", []))
        planned_counter = list(interpretation.get("counter_queries", []))
        scope = state["scope"]
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        competitors = scope.get("target_competitors", [])

        balanced_positive: list[str] = []
        balanced_counter: list[str] = []
        for technology in technologies:
            technology_term = self._technology_query_term(technology)
            for competitor in competitors:
                balanced_positive.append(
                    f"{technology_term} {competitor} official announcement roadmap sample production benchmark semiconductor memory 2025 2026"
                )
                balanced_counter.append(
                    f"{technology_term} {competitor} delay issue challenge limitation yield problem semiconductor memory 2024 2025 2026"
                )

        positive_queries = self._dedupe_strings(planned_web + balanced_positive)
        counter_queries = self._dedupe_strings(planned_counter + balanced_counter)
        positive_limit = max(1, self.config.max_web_queries // 2)
        counter_limit = max(1, self.config.max_web_queries - positive_limit)
        positive_queries = positive_queries[:positive_limit]
        counter_queries = counter_queries[:counter_limit]
        return positive_queries, counter_queries

    def _detect_web_search_failure_reason(
        self,
        *,
        source_diversity: int,
        freshness_score: float,
        has_counter_evidence: bool,
        source_reliability_score: float,
        bias_risk_score: float,
        balanced_company_coverage: bool,
    ) -> str:
        """수집된 웹 근거가 아직 충분하지 않은 이유를 분류한다."""
        if source_diversity < self.config.min_source_diversity:
            return "low_source_diversity"
        if freshness_score < self.config.min_recent_ratio:
            return "stale_results"
        if not has_counter_evidence:
            return "missing_counter_evidence"
        if source_reliability_score < self.config.min_source_reliability_score:
            return "low_source_reliability"
        if bias_risk_score > self.config.max_bias_risk_score:
            return "high_bias_risk"
        if not balanced_company_coverage:
            return "imbalanced_company_coverage"
        return "generic_web_search_failure"

    def _rewrite_web_queries(
        self,
        *,
        interpretation: dict[str, Any],
        reason: str,
    ) -> list[str]:
        """감지된 실패 이유에 맞춰 긍정 웹 검색 질의를 확장한다."""
        technologies = interpretation.get("target_technologies", [])
        competitors = interpretation.get("target_competitors", [])
        existing = list(interpretation.get("web_queries", []))
        rewritten: list[str] = []

        for technology in technologies:
            technology_term = self._technology_query_term(technology)
            for competitor in competitors:
                if reason == "stale_results":
                    rewritten.append(
                        f"{technology_term} {competitor} 2025 2026 latest announcement press release roadmap news semiconductor memory"
                    )
                elif reason == "low_source_diversity":
                    rewritten.extend(
                        [
                            f"{technology_term} {competitor} official site press release 2025",
                            f"{technology_term} {competitor} news analysis report 2025 semiconductor memory",
                            f"{technology_term} {competitor} conference presentation 2025",
                        ]
                    )
                elif reason == "low_source_reliability":
                    rewritten.extend(
                        [
                            f"{technology_term} {competitor} official announcement",
                            f"{technology_term} {competitor} IEEE JEDEC conference semiconductor memory",
                        ]
                    )
                elif reason == "imbalanced_company_coverage":
                    rewritten.append(
                        f"{technology_term} {competitor} roadmap sample production benchmark 2025 2026 semiconductor memory"
                    )
                else:
                    rewritten.append(
                        f"{technology_term} {competitor} latest announcement production roadmap 2025 2026 semiconductor memory"
                    )

        return self._dedupe_strings(existing + rewritten)

    def _rewrite_counter_queries(
        self,
        *,
        interpretation: dict[str, Any],
        reason: str,
    ) -> list[str]:
        """커버리지나 편향 검증이 실패했을 때 반증 질의를 확장한다."""
        technologies = interpretation.get("target_technologies", [])
        competitors = interpretation.get("target_competitors", [])
        existing = list(interpretation.get("counter_queries", []))
        rewritten: list[str] = []

        for technology in technologies:
            technology_term = self._technology_query_term(technology)
            for competitor in competitors:
                if reason in {"missing_counter_evidence", "high_bias_risk"}:
                    rewritten.extend(
                        [
                            f"{technology_term} {competitor} delay issue challenge limitation yield problem semiconductor memory 2024 2025 2026",
                            f"{technology_term} {competitor} risk bottleneck failure concern adoption issue semiconductor memory",
                        ]
                    )
                else:
                    rewritten.append(
                        f"{technology_term} {competitor} limitation delay issue challenge semiconductor memory"
                    )

        return self._dedupe_strings(existing + rewritten)

    def _search_web(self, query: str, stance: str, errors: list[str]) -> list[dict[str, Any]]:
        """Tavily를 호출하고 원본 검색 결과를 워크플로우 레코드로 정규화한다."""
        if self.search_tool is None:
            return []
        try:
            raw = self.search_tool.invoke(query)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return []

        if isinstance(raw, dict) and raw.get("error"):
            errors.append(str(raw["error"])[:200])
            return []

        results = raw.get("results", raw if isinstance(raw, list) else [])
        normalized: list[dict[str, Any]] = []
        for item in results:
            title = item.get("title") or item.get("url") or "Untitled"
            url = item.get("url")
            source = item.get("source") or self._extract_domain(url or "") or "web"
            reliability_tier, reliability_score = self._score_source_reliability(url, source)
            published_at = item.get("published_date") or item.get("date")
            content = item.get("content") or item.get("snippet") or ""
            normalized.append(
                {
                    "title": title,
                    "source": source,
                    "url": url,
                    "content": content,
                    "query": query,
                    "published_at": published_at,
                    "stance": stance,
                    "source_type": "web_search",
                    "is_recent": self._is_recent_result(title, content, published_at),
                    "source_reliability_tier": reliability_tier,
                    "source_reliability_score": reliability_score,
                }
            )
        return normalized

    def _matches_scope(self, item: dict[str, Any], state: WebSearchInput) -> bool:
        """웹 검색 결과가 기술/경쟁사 범위와 충분히 맞는지 필터링한다."""
        scope = state["scope"]
        technologies = scope.get("target_technologies") or ([scope["target_technology"]] if scope.get("target_technology") else [])
        competitors = scope.get("target_competitors", [])
        haystack = " ".join(
            [
                item.get("title", ""),
                item.get("content", ""),
                item.get("source", ""),
                item.get("url", ""),
                item.get("query", ""),
            ]
        ).lower()
        technology_match = any(self._contains_alias(haystack, self._technology_aliases(technology)) for technology in technologies)
        competitor_match = any(self._contains_alias(haystack, self._competitor_aliases(competitor)) for competitor in competitors)
        return technology_match and competitor_match

    def _score_source_reliability(self, url: str | None, source: str) -> tuple[str, float]:
        """출처 도메인을 기준으로 신뢰도 등급과 점수를 부여한다."""
        domain = self._extract_domain(url or "") or source.lower()
        official_domains = (
            "samsung.com",
            "skhynix.com",
            "micron.com",
            "nvidia.com",
            "amd.com",
            "intel.com",
        )
        standards_domains = ("jedec.org", "snia.org", "cxlconsortium.org")
        academic_domains = ("ieee.org", "acm.org", "springer.com", "nature.com", "arxiv.org")
        reputable_media_domains = (
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "nikkei.com",
            "tomshardware.com",
            "anandtech.com",
            "servethehome.com",
            "trendforce.com",
            "digitimes.com",
            "eejournal.com",
            "techpowerup.com",
            "businesstimes.com.sg",
            "eetimes.com",
            "storagereview.com",
            "chosun.com",
        )
        ecosystem_domains = ("computeexpresslink.org", "futurememorystorage.com")

        if any(domain.endswith(item) for item in official_domains):
            return "official", 1.0
        if any(domain.endswith(item) for item in standards_domains):
            return "standards", 0.95
        if any(domain.endswith(item) for item in ecosystem_domains):
            return "ecosystem", 0.85
        if any(domain.endswith(item) for item in academic_domains):
            return "academic", 0.9
        if any(domain.endswith(item) for item in reputable_media_domains):
            return "reputable_media", 0.8
        if domain:
            return "general_web", 0.6
        return "unknown", 0.4

    def _is_recent_result(self, title: str, content: str, published_at: str | None) -> bool:
        """현재 분석 기간 기준으로 검색 결과가 충분히 최신인지 추정한다."""
        current_year = datetime.now().year
        if published_at:
            try:
                normalized = published_at.replace("Z", "+00:00")
                year = datetime.fromisoformat(normalized).year
                return year >= current_year - 2
            except Exception:
                year = self._parse_year(published_at)
                if year is not None:
                    return year >= current_year - 2

        combined = " ".join([title, content[:300]])
        year = self._parse_year(combined)
        return year is not None and year >= current_year - 2

    @staticmethod
    def _compute_competitor_coverage(results: list[dict[str, Any]], competitors: list[str]) -> dict[str, int]:
        """각 목표 경쟁사가 검색 레코드에 몇 번 언급되는지 센다."""
        coverage = {competitor: 0 for competitor in competitors}
        for item in results:
            haystack = " ".join(
                [
                    item.get("title", ""),
                    item.get("content", ""),
                    item.get("source", ""),
                    item.get("url", ""),
                ]
            ).lower()
            for competitor in competitors:
                aliases = WebSearchService._competitor_aliases(competitor)
                if WebSearchService._contains_alias(haystack, aliases):
                    coverage[competitor] += 1
        return coverage

    @staticmethod
    def _compute_competitor_concentration(coverage: dict[str, int]) -> float:
        """가장 많이 언급된 경쟁사가 전체 언급에서 차지하는 비율을 반환한다."""
        total = sum(coverage.values())
        if total == 0:
            return 1.0
        return max(coverage.values()) / total

    @staticmethod
    def _is_company_coverage_balanced(coverage: dict[str, int], competitors: list[str]) -> bool:
        """검색 결과가 편향을 줄일 만큼 충분한 경쟁사를 다루는지 확인한다."""
        if not competitors:
            return False
        covered = [name for name, count in coverage.items() if count > 0]
        return len(covered) >= min(2, len(competitors))

    @staticmethod
    def _compute_freshness_score(results: list[dict[str, Any]]) -> float:
        """최신 검색 레코드의 비율을 계산한다."""
        if not results:
            return 0.0
        fresh = sum(1 for item in results if item.get("is_recent"))
        return round(fresh / len(results), 4)

    def _prioritize_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """최근성과 신뢰도를 우선해 supportive/counter 결과를 균형 있게 남긴다."""
        if not results:
            return []

        def sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
            """최근 결과와 신뢰도 높은 출처가 먼저 오도록 정렬 키를 만든다."""
            return (
                0 if item.get("is_recent") else 1,
                -float(item.get("source_reliability_score", 0.0)),
                item.get("source", ""),
            )

        cap_per_stance = max(self.config.min_web_results, 6)
        supportive = sorted(
            [item for item in results if item.get("stance") != "counter"],
            key=sort_key,
        )
        counter = sorted(
            [item for item in results if item.get("stance") == "counter"],
            key=sort_key,
        )
        selected = supportive[:cap_per_stance] + counter[:cap_per_stance]

        if len(selected) < max(self.config.min_web_results * 2, 12):
            selected_keys = {
                item.get("url") or f'{item.get("source", "")}:{item.get("title", "")}'
                for item in selected
            }
            remaining = [
                item for item in sorted(results, key=sort_key)
                if (item.get("url") or f'{item.get("source", "")}:{item.get("title", "")}') not in selected_keys
            ]
            needed = max(self.config.min_web_results * 2, 12) - len(selected)
            selected.extend(remaining[:needed])

        return selected

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
    def _extract_domain(url: str) -> str:
        """HTTP(S) URL에서 소문자 호스트 영역을 추출한다."""
        match = re.search(r"https?://([^/]+)", url or "")
        return match.group(1).lower() if match else ""

    @staticmethod
    def _parse_year(text: str) -> int | None:
        """텍스트에 20xx 연도가 있으면 첫 번째 연도를 추출한다."""
        match = re.search(r"\b(20\d{2})\b", text or "")
        return int(match.group(1)) if match else None

    @staticmethod
    def _contains_alias(text: str, aliases: list[str]) -> bool:
        """주어진 텍스트가 alias 후보 중 하나를 포함하는지 반환한다."""
        lowered = text.lower()
        return any(alias in lowered for alias in aliases)

    @staticmethod
    def _technology_query_term(technology: str) -> str:
        """웹 검색에서 모호성을 줄이기 위한 기술명 질의 표현을 반환한다."""
        mapping = {
            "HBM4": "\"HBM4\" \"high bandwidth memory\"",
            "HBM": "\"HBM\" \"high bandwidth memory\"",
            "PIM": "\"processing in memory\" PIM DRAM semiconductor",
            "CXL": "\"Compute Express Link\" CXL memory datacenter",
        }
        return mapping.get(technology, technology)

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
