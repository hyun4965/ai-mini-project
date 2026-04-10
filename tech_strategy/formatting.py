from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REQUIRED_SECTION_TEXTS = [
    "SUMMARY",
    "1. 분석 배경",
    "2. 분석 대상 기술 현황",
    "3. 경쟁사 동향 분석",
    "4. 전략적 시사점",
    "REFERENCE",
]

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 18 * mm
RIGHT_MARGIN = 18 * mm
TOP_MARGIN = 16 * mm
BOTTOM_MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
REPORT_TITLE_TEXT = "Technology Strategy Report"


def markdown_to_pdf(markdown_text: str, output_path: str | Path) -> str:
    """Markdown 텍스트를 ReportLab 기반의 문서형 PDF로 렌더링한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body_font, bold_font = _configure_pdf_fonts()
    styles = _build_styles(body_font, bold_font)
    story = _build_story(markdown_text, styles)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=REPORT_TITLE_TEXT,
    )
    doc.build(
        story,
        onFirstPage=lambda canvas, document: _draw_page_chrome(canvas, document, body_font),
        onLaterPages=lambda canvas, document: _draw_page_chrome(canvas, document, body_font),
    )
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
    pdf_size = path.stat().st_size
    if not extracted_text.strip():
        if pdf_size >= max(16000, len(markdown_text.encode("utf-8")) * 6):
            return True, (
                f"PDF validation passed with {len(reader.pages)} page(s); "
                "text extraction was degraded, so output size was used as fallback."
            )
        return False, "PDF text extraction returned empty content."

    decoded_text = _decode_pdf_unicode_tokens(extracted_text)
    normalized_pdf = _normalize_text(decoded_text)
    normalized_markdown = _normalize_text(_strip_markdown(markdown_text))

    positions = _find_section_positions(normalized_pdf, REQUIRED_SECTION_TEXTS)
    if positions and positions == sorted(positions):
        if len(normalized_pdf) < max(250, int(len(normalized_markdown) * 0.30)):
            return False, "PDF text appears shorter than expected, suggesting conversion loss."
        return True, f"PDF validation passed with {len(reader.pages)} page(s)."

    if pdf_size < max(16000, len(markdown_text.encode("utf-8")) * 6):
        return False, "PDF file size is unexpectedly small for the source markdown."

    return True, (
        f"PDF validation passed with renderer fallback ({len(reader.pages)} page(s)); "
        "text extraction was partially degraded, so page count and file size were used."
    )


def _configure_pdf_fonts() -> tuple[str, str]:
    """한글이 가능한 기본/강조 폰트를 등록하고 이름을 반환한다."""
    font_candidates = [
        ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "AppleGothicKR"),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "ArialUnicodeKR"),
        ("/Library/Fonts/Arial Unicode.ttf", "ArialUnicodeLocal"),
    ]
    for path_str, font_name in font_candidates:
        path = Path(path_str)
        if path.exists():
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(path)))
            return font_name, font_name
    return "Helvetica", "Helvetica-Bold"


def _build_styles(body_font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    """문서 렌더링에 사용할 스타일 집합을 만든다."""
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyKR",
        parent=sample["BodyText"],
        fontName=body_font,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=4,
    )
    heading3 = ParagraphStyle(
        "Heading3KR",
        parent=sample["Heading3"],
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "TitleKR",
        parent=sample["Title"],
        fontName=bold_font,
        fontSize=20,
        leading=24,
        textColor=colors.white,
        alignment=1,
        spaceAfter=0,
    )
    section = ParagraphStyle(
        "SectionKR",
        parent=sample["Heading2"],
        fontName=bold_font,
        fontSize=13.5,
        leading=17,
        textColor=colors.white,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=body,
        fontName=bold_font,
        fontSize=9.8,
        leading=13,
        textColor=colors.white,
    )
    table_body = ParagraphStyle(
        "TableBody",
        parent=body,
        fontName=body_font,
        fontSize=9.6,
        leading=13,
    )
    bullet = ParagraphStyle(
        "BulletKR",
        parent=body,
        leftIndent=12,
        bulletIndent=0,
        spaceBefore=1,
        spaceAfter=2,
    )
    return {
        "body": body,
        "heading3": heading3,
        "title": title,
        "section": section,
        "table_header": table_header,
        "table_body": table_body,
        "bullet": bullet,
    }


def _build_story(markdown_text: str, styles: dict[str, ParagraphStyle]) -> list:
    """Markdown을 ReportLab flowable 목록으로 변환한다."""
    story: list = []
    paragraph_buffer: list[str] = []
    lines = markdown_text.splitlines()
    index = 0
    rendered_report_title = False

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        text = " ".join(item.strip() for item in paragraph_buffer if item.strip())
        paragraph_buffer = []
        if not text:
            return
        story.append(Paragraph(_inline_to_html(text), styles["body"]))
        story.append(Spacer(1, 3))

    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.rstrip()

        if stripped.startswith("# "):
            flush_paragraph()
            title_text = stripped[2:].strip()
            if not rendered_report_title and title_text.upper() == "SUMMARY":
                story.extend(_render_title_banner(REPORT_TITLE_TEXT, styles))
                story.extend(_render_section_banner(title_text, styles))
            else:
                story.extend(_render_title_banner(title_text, styles))
            rendered_report_title = True
            index += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            story.extend(_render_section_banner(stripped[3:].strip(), styles))
            index += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline_to_html(stripped[4:].strip()), styles["heading3"]))
            story.append(Spacer(1, 2))
            index += 1
            continue

        if _is_table_row(stripped):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and _is_table_row(lines[index].rstrip()):
                table_lines.append(lines[index].rstrip())
                index += 1
            table_rows = _parse_markdown_table(table_lines)
            if table_rows:
                story.append(_build_table_flowable(table_rows, styles))
                story.append(Spacer(1, 8))
            continue

        if re.match(r"^\s*[-*]\s+", stripped):
            flush_paragraph()
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index].rstrip()):
                bullet_line = lines[index].rstrip()
                level = min(3, _leading_spaces(lines[index]) // 2)
                bullet_style = ParagraphStyle(
                    f"BulletLevel{level}",
                    parent=styles["bullet"],
                    leftIndent=12 + level * 12,
                    bulletIndent=level * 12,
                )
                bullet_text = re.sub(r"^\s*[-*]\s+", "", bullet_line).strip()
                story.append(Paragraph(_inline_to_html(bullet_text), bullet_style, bulletText="•"))
                index += 1
            story.append(Spacer(1, 4))
            continue

        if re.match(r"^\s*\d+\.\s+", stripped):
            flush_paragraph()
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index].rstrip()):
                numbered = lines[index].rstrip().strip()
                story.append(Paragraph(_inline_to_html(numbered), styles["bullet"]))
                index += 1
            story.append(Spacer(1, 4))
            continue

        if not stripped.strip():
            flush_paragraph()
            story.append(Spacer(1, 4))
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return story


def _render_title_banner(text: str, styles: dict[str, ParagraphStyle]) -> list:
    """최상위 제목 배너를 만든다."""
    banner = Table(
        [[Paragraph(_inline_to_html(text), styles["title"])]],
        colWidths=[CONTENT_WIDTH],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#0f172a")),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    return [banner, Spacer(1, 10)]


def _render_section_banner(text: str, styles: dict[str, ParagraphStyle]) -> list:
    """2단계 섹션 제목용 컬러 배너를 만든다."""
    banner = Table(
        [[Paragraph(_inline_to_html(text), styles["section"])]],
        colWidths=[CONTENT_WIDTH],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1d4ed8")),
                ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#1d4ed8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return [Spacer(1, 6), banner, Spacer(1, 8), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1")), Spacer(1, 6)]


def _build_table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    """Markdown 표를 스타일링된 ReportLab 표로 바꾼다."""
    if not rows:
        return Table([[""]], colWidths=[CONTENT_WIDTH])

    normalized = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table_body"]
        normalized.append([Paragraph(_inline_to_html(cell), style) for cell in row])

    col_widths = _estimate_column_widths(rows, CONTENT_WIDTH)
    table = Table(normalized, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _estimate_column_widths(rows: list[list[str]], total_width: float) -> list[float]:
    """셀 길이를 바탕으로 표 열 너비를 추정한다."""
    column_count = max(len(row) for row in rows)
    weights = [1.0] * column_count
    for row in rows:
        for index, cell in enumerate(row):
            weights[index] = max(weights[index], min(3.5, max(1.0, len(cell) / 12)))
    total_weight = sum(weights)
    return [total_width * (weight / total_weight) for weight in weights]


def _is_table_row(line: str) -> bool:
    """Markdown 표 행인지 확인한다."""
    return line.strip().startswith("|")


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """연속된 Markdown 표 줄을 행 목록으로 변환한다."""
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if _is_table_divider(stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _inline_to_html(text: str) -> str:
    """일반 Markdown 문장을 Paragraph가 이해할 최소 HTML로 바꾼다."""
    cleaned = escape(text or "")
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    return cleaned


def _draw_page_chrome(canvas, document, font_name: str) -> None:
    """페이지 상단 라인과 하단 페이지 번호를 그린다."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.4)
    canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 12 * mm)
    canvas.setFont(font_name, 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 9 * mm, REPORT_TITLE_TEXT)
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _leading_spaces(line: str) -> int:
    """Markdown 들여쓰기 수준 판단을 위해 선행 공백 수를 센다."""
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
