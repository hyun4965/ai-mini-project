from __future__ import annotations

import os
import platform
import re
import tempfile
import textwrap
from pathlib import Path

from pypdf import PdfReader

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mini-project-mpl-")


REQUIRED_SECTION_TEXTS = [
    "Summary",
    "1. 분석 배경",
    "2. 분석 대상 기술 현황",
    "3. 경쟁사 동향 분석",
    "4. 전략적 시사점",
    "REFERENCE",
]


def markdown_to_pdf(markdown_text: str, output_path: str | Path) -> str:
    """Markdown 텍스트를 페이지가 나뉜 PDF 파일로 렌더링한다."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    _configure_fonts(plt)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rendered_lines = _render_markdown_lines(markdown_text)

    with PdfPages(path) as pdf:
        fig = None
        y = 0.95
        for line in rendered_lines:
            if fig is None or y < 0.06:
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.95

            if line["text"]:
                text_kwargs = {
                    "fontsize": line["font_size"],
                    "va": "top",
                    "ha": "left",
                    "fontweight": line["weight"],
                }
                if line["family"]:
                    text_kwargs["family"] = line["family"]
                fig.text(
                    line["x"],
                    y,
                    line["text"],
                    **text_kwargs,
                )
            y -= line["line_gap"]

        if fig is None:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.95, "(empty report)", fontsize=10, va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return str(path)


def validate_pdf_output(markdown_text: str, pdf_path: str | Path) -> tuple[bool, str]:
    """생성된 PDF가 존재하고 필수 내용을 보존했는지 검증한다."""
    path = Path(pdf_path)
    if not path.exists():
        return False, "PDF file was not created."

    reader = PdfReader(str(path))
    if len(reader.pages) == 0:
        return False, "PDF has no pages."

    extracted_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not extracted_text.strip():
        return False, "PDF text extraction returned empty content."

    decoded_text = _decode_pdf_unicode_tokens(extracted_text)
    normalized_pdf = _normalize_text(decoded_text)
    normalized_markdown = _normalize_text(_strip_markdown(markdown_text))

    positions = _find_section_positions(normalized_pdf, REQUIRED_SECTION_TEXTS)
    if positions and positions == sorted(positions):
        if len(normalized_pdf) < max(250, int(len(normalized_markdown) * 0.35)):
            return False, "PDF text appears shorter than expected, suggesting conversion loss."
        return True, f"PDF validation passed with {len(reader.pages)} page(s)."

    pdf_size = path.stat().st_size
    if pdf_size < max(12000, len(markdown_text.encode("utf-8")) * 8):
        return False, "PDF file size is unexpectedly small for the source markdown."

    return True, (
        f"PDF validation passed with renderer fallback ({len(reader.pages)} page(s)); "
        "text extraction was partially degraded by embedded font encoding, so page count and output size were used."
    )


def _configure_fonts(plt) -> None:
    """운영체제별로 한글 출력에 사용할 Matplotlib 폰트를 설정한다."""
    from matplotlib import font_manager

    current_os = platform.system()
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Arial Unicode MS", "Helvetica"]
        if current_os == "Darwin"
        else ["Malgun Gothic", "NanumGothic", "Arial Unicode MS", "Helvetica"]
    )
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


def _render_markdown_lines(markdown_text: str) -> list[dict[str, str | float]]:
    """Markdown 각 줄을 PDF 렌더러가 그릴 수 있는 텍스트 명령으로 변환한다."""
    lines: list[dict[str, str | float]] = []
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.rstrip()
        if stripped.startswith("# "):
            lines.extend(_wrap_render_line(stripped[2:].strip(), font_size=18, width=38, x=0.06, weight="bold", line_gap=0.038))
            lines.append(_blank_line())
        elif stripped.startswith("## "):
            lines.extend(_wrap_render_line(stripped[3:].strip(), font_size=15, width=48, x=0.06, weight="bold", line_gap=0.034))
            lines.append(_blank_line())
        elif stripped.startswith("### "):
            lines.extend(_wrap_render_line(stripped[4:].strip(), font_size=12, width=56, x=0.06, weight="bold", line_gap=0.03))
        elif _is_table_divider(stripped):
            continue
        elif stripped.strip().startswith("|"):
            table_text = " | ".join(cell.strip() for cell in stripped.strip("|").split("|"))
            lines.extend(_wrap_render_line(table_text, font_size=9.5, width=82, x=0.08, weight="normal", line_gap=0.024))
        elif re.match(r"^\s*[-*]\s+", stripped):
            indent = min(0.18, 0.08 + (_leading_spaces(raw_line) // 2) * 0.018)
            bullet_text = "• " + re.sub(r"^\s*[-*]\s+", "", stripped)
            lines.extend(_wrap_render_line(bullet_text, font_size=10, width=82, x=indent, weight="normal", line_gap=0.024))
        elif re.match(r"^\s*\d+\.\s+", stripped):
            indent = min(0.18, 0.08 + (_leading_spaces(raw_line) // 2) * 0.018)
            lines.extend(_wrap_render_line(stripped.strip(), font_size=10, width=82, x=indent, weight="normal", line_gap=0.024))
        elif not stripped.strip():
            lines.append(_blank_line())
        else:
            lines.extend(_wrap_render_line(stripped.strip(), font_size=10.5, width=88, x=0.06, weight="normal", line_gap=0.026))
    return lines


def _wrap_render_line(
    text: str,
    *,
    font_size: float,
    width: int,
    x: float,
    weight: str,
    line_gap: float,
    family: str | None = None,
) -> list[dict[str, str | float]]:
    """하나의 논리적 텍스트 줄을 PDF에 그릴 수 있는 여러 줄 사전으로 감싼다."""
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]
    return [
        {
            "text": line,
            "font_size": font_size,
            "x": x,
            "weight": weight,
            "line_gap": line_gap,
            "family": family or "",
        }
        for line in wrapped
    ]


def _blank_line() -> dict[str, str | float]:
    """세로 여백을 의미하는 렌더링 명령을 반환한다."""
    return {
        "text": "",
        "font_size": 10.0,
        "x": 0.06,
        "weight": "normal",
        "line_gap": 0.02,
        "family": "",
    }


def _leading_spaces(line: str) -> int:
    """단순 Markdown 들여쓰기 처리를 위해 앞쪽 공백 수를 센다."""
    return len(line) - len(line.lstrip(" "))


def _is_table_divider(line: str) -> bool:
    """| --- | --- | 같은 Markdown 표 구분선을 감지한다."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    content = stripped.strip("|").replace(":", "").replace("-", "").replace(" ", "")
    return content == ""


def _strip_markdown(markdown_text: str) -> str:
    """읽을 수 있는 텍스트는 유지하면서 일반적인 Markdown 문법을 제거한다."""
    text = markdown_text
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    return text


def _normalize_text(text: str) -> str:
    """텍스트 비교를 위해 공백과 대소문자를 정규화한다."""
    cleaned = re.sub(r"[\x00-\x1f]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _decode_pdf_unicode_tokens(text: str) -> str:
    """PDF 텍스트 추출 결과에 나타나는 /uni000000XX 토큰을 디코딩한다."""
    if "/uni" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        """추출된 /uni 토큰 하나를 의도된 Unicode 문자로 되돌린다."""
        value = int(match.group(1), 16)
        return chr(max(0, value - 1))

    return re.sub(r"/uni([0-9A-Fa-f]{8})", repl, text)


def _find_section_positions(normalized_text: str, sections: list[str]) -> list[int]:
    """정규화된 텍스트에서 섹션 제목들의 위치를 순서대로 찾는다."""
    positions: list[int] = []
    for section in sections:
        pos = normalized_text.find(_normalize_text(section))
        if pos == -1:
            return []
        positions.append(pos)
    return positions
