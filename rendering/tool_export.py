from __future__ import annotations

from ui.display_payloads import build_tool_result_display_data


def build_tool_result_markdown_lines(tool_result: object) -> list[str]:
    tool_display = build_tool_result_display_data(tool_result)
    if tool_display is None:
        return []

    lines = [f"- Tool: {tool_display['tool_name']}"]
    if tool_display["raw_query"] is not None:
        lines.append(f"- Original query: {tool_display['raw_query']}")
    if tool_display["input_fields"]:
        lines.append("- Input:")
        lines.extend(_build_markdown_tool_field_lines(tool_display["input_fields"]))
    if tool_display["output_fields"]:
        lines.append("- Result:")
        lines.extend(_build_markdown_tool_field_lines(tool_display["output_fields"]))
    if tool_display["error"] is not None:
        lines.append(f"- Error: {tool_display['error']}")
    return lines


def _build_markdown_tool_field_lines(fields: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for field in fields:
        label = field["label"]
        field_lines = field["lines"]
        if len(field_lines) == 1:
            lines.append(f"  - {label}: {field_lines[0]}")
            continue
        lines.append(f"  - {label}:")
        for item in field_lines:
            lines.append(f"    - {item}")
    return lines


def build_tool_result_text_lines(tool_result: object) -> list[str]:
    tool_display = build_tool_result_display_data(tool_result)
    if tool_display is None:
        return []

    lines = [f"Tool result: {tool_display['tool_name']}"]
    if tool_display["raw_query"] is not None:
        lines.append(f"- Original query: {tool_display['raw_query']}")
    if tool_display["input_fields"]:
        lines.append("Input:")
        lines.extend(_build_text_tool_field_lines(tool_display["input_fields"]))
    if tool_display["output_fields"]:
        lines.append("Result:")
        lines.extend(_build_text_tool_field_lines(tool_display["output_fields"]))
    if tool_display["error"] is not None:
        lines.append(f"- Error: {tool_display['error']}")
    return lines


def _build_text_tool_field_lines(fields: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for field in fields:
        label = field["label"]
        field_lines = field["lines"]
        if len(field_lines) == 1:
            lines.append(f"- {label}: {field_lines[0]}")
            continue
        lines.append(f"{label}:")
        for item in field_lines:
            lines.append(f"- {item}")
    return lines
