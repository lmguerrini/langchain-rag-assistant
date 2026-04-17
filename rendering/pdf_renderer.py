from __future__ import annotations

import json
import re

from fpdf import FPDF

from rendering.response_labels import get_response_type_label
from rendering.tool_export import build_tool_result_text_lines


PDF_TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u200b": "",
        "\u2013": " ",
        "\u2014": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
    }
)


def build_conversation_pdf(conversation_history: list[dict[str, object]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_title(normalize_text_for_pdf("Conversation Export"))

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 10, normalize_text_for_pdf("Conversation Export"))
    pdf.ln(12)
    pdf.set_font("Helvetica", size=11)

    if not conversation_history:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, normalize_text_for_pdf("No conversation history available."))
    else:
        for index, turn in enumerate(conversation_history, start=1):
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 8, normalize_text_for_pdf(f"Turn {index}"))
            pdf.ln(8)
            pdf.set_font("Helvetica", size=11)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, normalize_text_for_pdf(f"User question: {turn['query']}"))
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                7,
                normalize_text_for_pdf(
                    f"Response type: {get_response_type_label(turn)}"
                ),
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                7,
                normalize_text_for_pdf(
                    f"Assistant answer: {clean_markdown_text_for_pdf(turn['answer'])}"
                ),
            )

            if turn["sources"]:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 7, normalize_text_for_pdf("Sources:"))
                for source in turn["sources"]:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(f"- {source}"))

            if turn["tool_result"]:
                for line in build_tool_result_text_lines(turn["tool_result"]):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(line))

            if turn["usage"]:
                for line in build_pdf_detail_lines("Usage", turn["usage"]):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(line))

            pdf.ln(3)

    rendered = pdf.output()
    if isinstance(rendered, bytearray):
        return bytes(rendered)
    if isinstance(rendered, bytes):
        return rendered
    return rendered.encode("latin-1")


def normalize_text_for_pdf(text: object) -> str:
    normalized = str(text).translate(PDF_TEXT_REPLACEMENTS)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def clean_markdown_text_for_pdf(text: object) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw_line)
        line = line.replace("**", "")
        if re.fullmatch(r"\s*-{3,}\s*", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def build_pdf_detail_lines(title: str, data: dict[str, object]) -> list[str]:
    lines = [f"{title}:"]
    for key, value in data.items():
        if key == "estimated_cost_usd":
            label = "Estimated cost USD"
            value_text = (
                f"{value:.6f}"
                if isinstance(value, int | float)
                else "N/A"
            )
        else:
            label = key.replace("_", " ").capitalize()
            if isinstance(value, dict | list):
                value_text = json.dumps(value, ensure_ascii=False)
            elif value is None:
                value_text = "none"
            else:
                value_text = str(value)
        lines.append(f"- {label}: {value_text}")
    return lines
