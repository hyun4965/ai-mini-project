from __future__ import annotations


class TechStrategyError(Exception):
    """기술 전략 워크플로우 전반에서 사용하는 기본 예외 타입."""


class ExternalServiceError(TechStrategyError):
    """외부 서비스 호출 실패를 나타내는 공통 예외."""

    def __init__(self, service: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.service = service
        self.retryable = retryable


class LLMServiceError(ExternalServiceError):
    """LLM 호출 실패."""


class WebSearchServiceError(ExternalServiceError):
    """웹 검색 서비스 호출 실패."""


class ServiceTimeoutError(ExternalServiceError):
    """외부 서비스 타임아웃."""

    def __init__(self, service: str, timeout_seconds: float) -> None:
        super().__init__(
            service,
            f"{service} request timed out after {timeout_seconds:.1f}s",
            retryable=True,
        )
        self.timeout_seconds = timeout_seconds


class WorkflowExecutionTimeoutError(TechStrategyError):
    """워크플로우 전체 실행 타임아웃."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"workflow execution timed out after {timeout_seconds:.1f}s")
        self.timeout_seconds = timeout_seconds


class DocumentLoadError(TechStrategyError):
    """문서 로드 또는 파싱 실패."""


class EmbeddingInitializationError(TechStrategyError):
    """임베딩 모델 초기화 실패."""


class VectorStoreError(TechStrategyError):
    """벡터 스토어 로드/생성 실패."""


class FormattingError(TechStrategyError):
    """출력 포맷팅 실패."""


class OutputWriteError(FormattingError):
    """산출물 파일 저장 실패."""


class PDFValidationError(FormattingError):
    """PDF 검증 실패."""

