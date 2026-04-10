from __future__ import annotations

import argparse
from pathlib import Path

from .config import StrategyConfig, load_project_env
from .formatting import markdown_to_pdf, validate_pdf_output


def build_report_template(config: StrategyConfig) -> str:
    """실제 워크플로우 실행 전 제출 형식을 확인할 수 있는 기본 보고서 Markdown을 만든다."""
    return """# SUMMARY

이 문서는 실데이터 실행 전 제출 형식을 맞추기 위한 기본 산출물 템플릿이다. 실제 보고서 실행 시 Retrieval, Web Search, Assessment, Decision 결과가 채워지며, HBM / PIM / CXL에 대한 경쟁사 비교와 R&D 우선순위 판단이 반영된다.

- 분석 대상 기술(HBM, PIM, CXL)의 현재 기술 수준 요약
- 경쟁사별 기술 성숙도(TRL) 및 위협 수준 비교
- R&D 관점에서의 핵심 대응 방향 및 우선순위 제시
- TRL 4~6 구간에 대한 평가 한계 및 불확실성 명시

## 1. 분석 배경

### 1.1 분석 목적

해당 기술을 왜 지금 분석해야 하는지, AI와 데이터센터 시장 변화가 메모리 및 인터커넥트 기술 경쟁을 어떻게 자극하는지, 그리고 R&D 의사결정 지원 관점에서 왜 비교 분석이 필요한지를 설명하는 영역이다.

### 1.2 분석 범위 및 기준

- 분석 대상 기술: HBM, PIM, CXL
- 기준 기업: SK hynix
- 분석 대상 경쟁사: Samsung, Micron 등 주요 메모리 및 시스템 반도체 기업
- 활용 데이터 범위:
  - 공개 자료 기반 (논문, 기업 발표, 기사 등)
  - 내부 정보는 포함하지 않음

### 1.3 TRL 기반 평가 기준 정의

TRL은 NASA가 우주 기술의 개발 단계를 표준화하기 위해 만든 9단계 기술 성숙도 척도를 반도체 기술 분석에 적용한 기준이다. SWOT처럼 강하다/약하다에 가까운 상대적 인상에 머무르지 않고, 각 기술이 지금 몇 단계에 있는지라는 절대적 위치를 제시하기 위해 사용한다.

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

실행 시 HBM 계열 기술 개요, 고대역폭 메모리 구조 및 성능 특징, 현재 개발 단계와 발전 방향이 채워진다. HBM4가 포함된 경우 HBM 내 차세대 세부 트랙으로 함께 해석한다.

### 2.2 PIM 기술 현황

실행 시 메모리 내 연산 구조, 활용 분야, 현재 개발 단계와 발전 방향이 채워진다.

### 2.3 CXL 기술 현황

실행 시 메모리 확장과 인터페이스 구조, 시스템 아키텍처 변화와의 연관성, 현재 개발 단계와 발전 방향이 채워진다.

## 3. 경쟁사 동향 분석

### 3.1 경쟁사별 기술 개발 방향

실행 시 Samsung, SK hynix, Micron 등 주요 경쟁사의 기술 전략과 기술별 집중 영역 및 방향성 비교가 채워진다.

### 3.2 TRL 기반 기술 성숙도 비교

실행 시 기술별 TRL 수준 비교와 경쟁사 간 상대적 기술 위치 분석이 채워진다.

### 3.3 위협 수준 평가

실행 시 기술 완성도, 시장 영향력, 경쟁 강도를 바탕으로 High / Medium / Low 위협 수준 분류가 채워진다.

## 4. 전략적 시사점

### 4.1 기술별 전략적 중요도

실행 시 기술별 R&D 추진 필요성, 투자 우선순위(High / Medium / Low), 판단 근거(TRL, 경쟁사, 시장 영향)가 채워진다.

### 4.2 경쟁 대응 방향

실행 시 경쟁사 대비 추격 전략, 차별화 전략, 협력 전략이 기술별 판단과 연결되어 채워진다.

### 4.3 한계

- TRL 4~6 구간은 공개 정보 기반 추정에 해당함
- TRL 4 이상을 정확히 판정하려면 내부 문서, 통합 검증 기록, 공정 및 수율 데이터, 고객 샘플 검증 자료가 필요하지만 본 workflow는 공개 자료만 사용함
- 수율, 공정, 성능 데이터는 비공개 영역으로 정확한 평가에 한계 존재
- 특허, 발표, 채용 등의 간접 지표를 기반으로 판단 수행

## REFERENCE

- 논문 및 학회 발표 자료
- 기업 공식 발표 및 보도자료
- 산업 분석 리포트
- 뉴스 및 기사 자료
"""


def generate_report_template(config: StrategyConfig) -> tuple[Path, Path]:
    """보고서 템플릿 Markdown과 PDF를 생성하고 PDF 변환 결과를 검증한다."""
    markdown = build_report_template(config)
    md_path = config.output_dir / f"ai-mini_output_{config.deliverable_label}.md"
    pdf_path = config.output_dir / f"ai-mini_output_{config.deliverable_label}.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    generated_path = markdown_to_pdf(markdown, pdf_path)
    is_valid, message = validate_pdf_output(markdown, generated_path)
    if not is_valid:
        raise RuntimeError(message)
    return md_path, pdf_path


def parse_args() -> argparse.Namespace:
    """보고서 템플릿 생성 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Generate a submission-ready report template PDF.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root for mini_project",
    )
    parser.add_argument(
        "--team-label",
        default=None,
        help="Deliverable label, for example 3반_배석현+박나연",
    )
    return parser.parse_args()


def main() -> None:
    """환경 설정을 로드한 뒤 보고서 템플릿 파일을 생성하는 CLI 진입점."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    load_project_env(project_root)
    config = StrategyConfig.from_project_root(project_root)
    if args.team_label:
        config.deliverable_label = args.team_label
    md_path, pdf_path = generate_report_template(config)
    print(md_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
