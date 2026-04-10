from __future__ import annotations

import argparse
from pathlib import Path

from .config import StrategyConfig, load_project_env
from .formatting import markdown_to_pdf, validate_pdf_output


def build_design_markdown(config: StrategyConfig) -> str:
    """워크플로우 설계 산출물에 들어갈 Markdown 본문을 생성한다."""
    label = config.deliverable_label
    return f"""# AI Mini Design

Team Label: {label}

## 1. Workflow

### Goal

본 시스템의 목표는 HBM4, PIM, CXL과 같은 차세대 메모리/인터커넥트 기술에 대해 경쟁사 분석, TRL 평가, 위협 수준 평가를 통합하고, 이를 바탕으로 SK hynix의 R&D 추진 타당성과 우선순위를 제안하는 기술 전략 분석 보고서를 생성하는 것이다.

### Success Criteria

- 신뢰성: Retrieval과 Web Search가 다중 출처를 확보하고 직접 근거와 간접 지표를 구분해야 한다.
- 정확성: Query와 높은 의미적 관련성을 갖는 자료를 수집하고 TRL/Threat를 일관된 기준으로 평가해야 한다.
- 설명가능성: TRL, Threat, Decision의 판단 근거가 추적 가능해야 한다.
- 의사결정 유효성: 단순 요약이 아니라 Go / Hold / Monitor와 Priority를 제시해야 한다.

### Task

1. 사용자 시나리오에서 기술과 경쟁사 범위를 해석한다.
2. Retrieval로 기본 지식층을 확보한다.
3. Web Search로 최신 동향과 반증 정보를 수집한다.
4. Assessment에서 evidence synthesis, TRL, Threat를 통합 평가한다.
5. Decision에서 R&D 추진 여부와 우선순위를 결정한다.
6. Draft에서 보고서를 작성한다.
7. Supervisor가 Draft를 검증한 뒤 Formatting Node에 PDF 생성을 요청한다.

### Control Strategy

- Supervisor는 흐름 제어, 검증, 재시도, 종료 판단만 수행한다.
- 각 Agent는 도메인 결과만 생성하고 종료를 결정하지 않는다.
- Retrieval / Web Search Agent는 내부에서 실패 원인을 진단하고 query rewrite를 수행한다.
- Retrieval / Web Search / Assessment / Draft는 반복 수행 가능하다.
- Draft는 절대 END로 직접 연결하지 않고 반드시 Supervisor로 복귀한다.
- Formatting은 PDF 생성만 수행하고 success/fail을 Supervisor에 반환한다.
- 재검색은 동일 query 반복이 아니라, 각 Agent가 실패 원인에 따라 query를 재구성하여 수행한다.
- 관련성 부족 시 기술 세부 키워드와 문서 유형 키워드를 추가한다.
- 편향 완화를 위해 반증 query를 별도로 확장한다.
- 최신성 부족 시 최근 연도, 발표, 보도자료 키워드를 강화한다.
- Decision은 TRL, Threat, competitor position, evidence quality와 연결된 근거가 없으면 통과하지 않는다.
- Decision 형식 누락은 Decision을 재실행하고, 근거 부족은 Assessment 단계로 재귀한다.
- Draft는 필수 섹션, Decision 반영, 근거 연결, TRL 4~6 한계 문구를 만족해야 통과한다.
- Draft가 단순 bullet 나열 중심이면 fallback 분석형 초안으로 다시 생성한다.
- Formatting은 markdown을 PDF로 변환한 뒤, PDF 텍스트 추출을 통해 섹션 순서와 내용 손실 여부를 다시 검증한다.
- Formatting 검증 실패는 success가 아니라 fail로 반환되며 Supervisor가 재시도 또는 오류 종료를 판단한다.

## 2. Structure Choice

### Selected Pattern: Supervisor

이 프로젝트는 Distributed보다 Supervisor를 선택했다.

이유:

- 검색 -> 분석 -> 의사결정 -> 보고서 생성의 단계 의존성이 강하다.
- 결과 품질에 따라 이전 단계로 되돌아가 보완해야 한다.
- 과제에서 가장 강조한 Draft 조기 종료 방지 요구를 가장 명확하게 만족한다.
- Supervisor의 R&R을 “부장님 역할”로 분리해, 각 Agent는 실무만 하고 승인/종료는 하지 않도록 설계할 수 있다.

### Architecture Flow

```mermaid
flowchart TD
    U[User Query] --> S[Supervisor]
    S --> R[Retrieval Agent]
    S --> W[Web Search Agent]
    R --> S
    W --> S
    S --> A[Assessment Agent]
    A --> S
    S --> D[Decision Agent]
    D --> S
    S --> G[Draft Agent]
    G --> S
    S -->|검증 OK| F[Formatting Node]
    F -->|success or fail| S
    S --> END[END]
```

ASCII flow:

```text
User Query
   -> Supervisor
      -> Retrieval Agent -> Supervisor
      -> Web Search Agent -> Supervisor
      -> Assessment Agent -> Supervisor
      -> Decision Agent -> Supervisor
      -> Draft Agent -> Supervisor
      -> Formatting Node -> Supervisor
      -> END
```

### Supervisor R&R

- 현재 단계 판정
- 정보 충분성 검증
- Draft 품질 검증
- Formatting 성공 여부 검증
- 재시도 횟수 관리
- 최종 종료 판단

## 3. Retrieve Design

### Open-Source Embedding Candidates

1. `intfloat/multilingual-e5-large`
2. `BAAI/bge-m3`
3. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### Embedding Selection Criteria

- 한국어와 영어가 섞인 반도체 기술 문서를 모두 잘 처리하는가
- 기술명, 기업명, 규격명(HBM4, CXL, Micron, JEDEC)을 semantic space에서 안정적으로 보존하는가
- PDF chunk 길이가 길어도 의미 손실이 적은가
- CPU 환경에서도 실무적으로 사용할 수 있는가
- 오픈소스 라이선스와 재현성이 확보되는가

### Final Embedding Choice

- 최종 선택: `intfloat/multilingual-e5-large`
- 선택 이유: 다국어 semantic retrieval 성능이 안정적이고 기술 키워드 + 자연어 문장 혼합 환경에서 강하다.
- 실행 안정성 확보를 위해 모델 로드 실패 시 lexical fallback을 허용했다.

### Retrieval Technique Candidates

1. Dense Similarity
2. BM25 / Korean lexical retrieval
3. Hybrid Retrieval (Dense + Lexical)
4. MMR
5. MultiQuery
6. Parent Document Retrieval

### Retrieval Selection Criteria

- Hit Rate@K
- MRR
- 한국어 기술 키워드 정확 매칭 성능
- 의미적 유사 문서 검색 성능
- 중복 문서 억제 능력
- 구현 복잡도와 운영 비용

### Final Retrieval Choice

- 최종 선택: Hybrid Retrieval
- 현재 구현: dense score + lexical overlap score의 가중 결합
- 선택 이유: 기술명/기업명 exact match와 semantic similarity를 동시에 반영하기 때문

### Retrieval Evaluation Plan

- 참고 기준: `langchain-v1/14-Retriever/10-Retriever-Evaluation.ipynb`
- README에는 최종 선정 Retrieval의 `Hit Rate@K`, `MRR`를 반영한다.
- 현재 저장소에는 evaluation scaffold를 포함했고, 실제 domain corpus가 채워지면 동일 방식으로 수치를 산출한다.

## 4. Web Search Bias Mitigation

- 긍정 query와 반증 query를 동시에 생성한다.
- 특정 기업 기사만 과대표집되지 않도록 source diversity를 측정한다.
- counter evidence가 없으면 Web Search Agent가 반증 query를 확장하고 Supervisor가 재수행을 승인한다.
- bias risk score를 계산해 특정 도메인 편향을 감시한다.
- 공식 사이트, 표준 단체, 학술 자료, 평판 있는 매체를 구분해 출처 신뢰도 점수를 계산한다.
- 최종 보고서에는 직접 근거와 간접 지표를 분리해서 서술한다.

## 5. Agent Definitions

### Retrieval Agent

- 역할: 로컬 문서 corpus에서 기본 지식층 검색
- 입력: retrieval queries
- 출력: filtered docs, relevance score, retrieval confidence

### Web Search Agent

- 역할: 최신 뉴스, 발표, 보도자료, 반증 정보 수집
- 입력: recent queries + counter queries
- 출력: news results, source diversity, freshness, bias indicators

### Assessment Agent

- 역할: evidence synthesis + TRL + Threat
- 내부 단계:
  - Stage 1: Direct vs Indirect Evidence Synthesis
  - Stage 2: TRL Assessment
  - Stage 3: Threat Assessment

### Decision Agent

- 역할: Go / Hold / Monitor, Priority 도출
- 기준: TRL, Threat, evidence quality, competitor pressure

### Draft Agent

- 역할: 보고서 초안 생성
- 주의: END를 직접 결정하지 않음

### Formatting Node

- 역할: Markdown -> PDF 변환
- 주의: Node이며, 품질 판단이나 종료 판단을 하지 않음

## 6. State Design

### Global State

- user_query
- scope: target_technology / target_technologies / target_competitors
- query_plan
- retrieval: retrieved_docs / filtered_docs / confidence / is_success / attempt
- web_search: web_results / source_diversity / freshness / bias indicators / is_success / attempt
- assessment: evidence_bundle / results / is_complete
- decision: result / is_valid / failure_reason
- draft: markdown_text / quality_score / is_valid
- output: pdf_path / final_pdf_path / is_pdf_generated / format_error
- analysis_log

### Supervisor State

- control: workflow_stage / coverage_status / is_information_sufficient
- control: status / retry_count / max_iteration
- control: next_step / final_decision / query_rewrite_history

## 7. Report Outline

- SUMMARY
- 1. 분석 배경
- 2. 분석 대상 기술 현황
- 3. 경쟁사 동향 분석
- 4. 전략적 시사점
- REFERENCE

## 8. TRL Interpretation and Explicit Limitation

- TRL 1~3은 공개 정보로 비교적 관찰 가능하다.
- TRL 4~6은 가장 큰 정보 공백 구간이며, 실제 수율/공정/성능 수치는 비공개일 가능성이 높다.
- TRL 7~9는 샘플 공급, 양산 발표, 실적 공시 등으로 일부 확인 가능하다.

본 프로젝트는 이 한계를 보고서에 명시적으로 적는다.

필수 문장 원칙:

- “TRL 4~6 구간은 공개 정보 기반 추정에 해당한다.”
- “TRL 4 이상을 정확히 판정하려면 내부 문서, 통합 검증 기록, 공정/수율 데이터, 고객 샘플 검증 자료가 필요하지만 본 workflow에는 해당 내부 자료가 없다.”
- “특허 출원 패턴, 학회 발표 빈도, 채용 공고 키워드와 같은 간접 지표를 근거로 추정했다.”

## 9. Output Naming

- 설계 산출물: `ai-mini_design_{label}.pdf`
- 개발 산출물: `ai-mini_output_{label}.pdf`
"""


def generate_design_artifact(config: StrategyConfig) -> tuple[Path, Path]:
    """설계 산출물 Markdown과 PDF를 생성하고 PDF 변환 결과를 검증한다."""
    markdown = build_design_markdown(config)
    md_path = config.output_dir / f"ai-mini_design_{config.deliverable_label}.md"
    pdf_path = config.output_dir / f"ai-mini_design_{config.deliverable_label}.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    generated_path = markdown_to_pdf(markdown, pdf_path)
    is_valid, message = validate_pdf_output(markdown, generated_path)
    if not is_valid:
        raise RuntimeError(message)
    return md_path, pdf_path


def parse_args() -> argparse.Namespace:
    """설계 산출물 생성 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Generate the design artifact PDF.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root for mini_project",
    )
    parser.add_argument(
        "--team-label",
        default=None,
        help="Deliverable label, for example 3반_김철수+최영희+박민수+이서연",
    )
    return parser.parse_args()


def main() -> None:
    """환경 설정을 로드한 뒤 설계 산출물 파일을 생성하는 CLI 진입점."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    load_project_env(project_root)
    config = StrategyConfig.from_project_root(project_root)
    if args.team_label:
        config.deliverable_label = args.team_label
    md_path, pdf_path = generate_design_artifact(config)
    print(md_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
