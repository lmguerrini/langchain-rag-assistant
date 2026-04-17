from __future__ import annotations

import altair as alt


def _format_cost_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"${value:.6f}"
    return "n/a"


def build_response_behavior_chart_rows(
    response_breakdown: list[dict[str, int | float | str]],
) -> list[dict[str, int | str]]:
    return [
        {
            "response_type": str(row["response_type"]),
            "count": int(row["count"]),
        }
        for row in response_breakdown
    ]


def build_response_behavior_chart(
    response_breakdown: list[dict[str, int | float | str]],
):
    return build_horizontal_bar_chart(
        chart_rows=build_response_behavior_chart_rows(response_breakdown),
        category_field="response_type",
        value_field="count",
        category_title="Response type",
        value_title="Count",
    )


def build_model_usage_chart_rows(
    model_usage_rows: list[dict[str, int | float | str | None]],
) -> list[dict[str, int | str]]:
    return [
        {
            "model": str(row["model"]),
            "total_tokens": int(row["total_tokens"]),
        }
        for row in model_usage_rows
    ]


def build_model_usage_chart(
    model_usage_rows: list[dict[str, int | float | str | None]],
):
    return build_horizontal_bar_chart(
        chart_rows=build_model_usage_chart_rows(model_usage_rows),
        category_field="model",
        value_field="total_tokens",
        category_title="Model",
        value_title="Total tokens",
    )


def build_horizontal_bar_chart(
    *,
    chart_rows: list[dict[str, int | str]],
    category_field: str,
    value_field: str,
    category_title: str,
    value_title: str,
):
    return (
        alt.Chart(alt.Data(values=chart_rows))
        .mark_bar()
        .encode(
            x=alt.X(f"{value_field}:Q", title=value_title),
            y=alt.Y(
                f"{category_field}:N",
                title=category_title,
                sort="-x",
                axis=alt.Axis(labelLimit=400),
            ),
            tooltip=[
                alt.Tooltip(f"{category_field}:N", title=category_title),
                alt.Tooltip(f"{value_field}:Q", title=value_title),
            ],
        )
        .properties(height=max(160, len(chart_rows) * 40))
    )
