from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


def df_to_table_data(df, max_rows=15):
    if df is None or getattr(df, "empty", True):
        return [["No data available"]]

    df = df.head(max_rows).copy()
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    return [headers] + rows


def make_table(df, max_rows=15):
    data = df_to_table_data(df, max_rows=max_rows)

    if len(data) == 1 and len(data[0]) == 1:
        table = Table(data, colWidths=[16 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e6f2ff")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#b3d7ff")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#FDBA74")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    ncols = len(data[0])
    col_width = 16 * cm / max(ncols, 1)
    table = Table(data, colWidths=[col_width] * ncols)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b3d7ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BFDBFE")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#EFF6FF"),
            colors.white
        ]),
    ]))
    return table


def fig_to_rl_image(fig, width_cm=16, height_cm=9):
    if fig is None:
        return None

    try:
        img_bytes = fig.to_image(format="png", scale=2)
        img_buffer = BytesIO(img_bytes)
        return Image(img_buffer, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None


def metric_table(report_data):
    data = [
        ["Original Rows", str(report_data.get("original_count", 0))],
        ["Cleaned Rows", str(report_data.get("cleaned_count", 0))],
        ["Removed Rows", str(report_data.get("removed_count", 0))],
        ["Dropped Columns", str(report_data.get("dropped_column_count", 0))],
    ]

    table = Table(data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#7C3AED")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F5F3FF")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#4C1D95")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C4B5FD")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDD6FE")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def create_full_report_pdf(report_data):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#b3d7ff"),
        borderPadding=10,
        borderRadius=8,
        spaceAfter=10,
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
        alignment=1,
    )

    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.white,
        backColor=colors.HexColor("#b3d7ff"),
        borderPadding=6,
        spaceBefore=12,
        spaceAfter=8,
    )

    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#b3d7ff"),
        borderPadding=5,
        spaceBefore=8,
        spaceAfter=6,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    note = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#92400E"),
        backColor=colors.HexColor("#b3d7ff"),
        borderPadding=5,
        spaceAfter=6,
    )

    badge = ParagraphStyle(
        "Badge",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#065F46"),
        backColor=colors.HexColor("#b3d7ff"),
        borderPadding=4,
        spaceAfter=6,
    )

    story = []

    # Cover
    story.append(Paragraph("AI Data Analysis Platform Report", title_style))
    story.append(Paragraph("Exported summary of user inputs, tables, and chart outputs.", subtitle_style))
    story.append(Paragraph("Generated Report", badge))
    story.append(Spacer(1, 8))

    # Overview
    story.append(Paragraph("Overview", h1))
    story.append(metric_table(report_data))
    story.append(Spacer(1, 10))

    # Cleaning Summary
    story.append(Paragraph("Cleaning Summary", h1))
    dropped_cols = report_data.get("columns_to_drop", [])
    remaining_cols = report_data.get("remaining_columns", [])

    story.append(Paragraph(
        f"<b>Dropped Columns:</b> {', '.join(dropped_cols) if dropped_cols else 'None'}",
        body
    ))
    story.append(Paragraph(
        f"<b>Remaining Columns:</b> {', '.join(remaining_cols) if remaining_cols else 'None'}",
        body
    ))
    story.append(Spacer(1, 8))

    # Filter blocks
    story.append(Paragraph("Filtered Data Blocks", h1))
    filter_blocks = report_data.get("filter_blocks", [])
    filter_figures = report_data.get("filter_figures", [])

    if not filter_blocks:
        story.append(Paragraph("No filtered blocks available.", note))
    else:
        for idx, block in enumerate(filter_blocks):
            story.append(Paragraph(block.get("name", f"Filtered Data Block {idx+1}"), h2))

            applied_filters = block.get("applied_filters", [])
            if applied_filters:
                filters_html = "<br/>".join([f"{col} = {val}" for col, val in applied_filters])
            else:
                filters_html = "No filters selected."

            story.append(Paragraph(f"<b>Applied Filters:</b><br/>{filters_html}", body))

            kpis = block.get("kpis", {})
            if kpis:
                kpi_html = "<br/>".join([f"{k}: {v}" for k, v in kpis.items()])
                story.append(Paragraph(f"<b>KPI Summary:</b><br/>{kpi_html}", body))

            story.append(Paragraph("<b>Filtered Result Preview:</b>", body))
            story.append(make_table(block.get("dataframe"), max_rows=10))
            story.append(Spacer(1, 6))

            matching_fig = next((f for f in filter_figures if f["title"] == block.get("name")), None)
            if matching_fig:
                img = fig_to_rl_image(matching_fig["figure"], width_cm=16, height_cm=8.5)
                if img:
                    story.append(Paragraph("<b>Chart:</b>", body))
                    story.append(img)
                    story.append(Spacer(1, 8))
                else:
                    story.append(Paragraph("Chart image skipped because Plotly image export is unavailable on this system.", note))
                    story.append(Spacer(1, 4))

    story.append(PageBreak())

    # Insight section
    story.append(Paragraph("Insight Output", h1))
    insight = report_data.get("insight_output")

    if insight:
        story.append(Paragraph(
            f"<b>Analysis Column:</b> {insight.get('analysis_column', 'N/A')}<br/>"
            f"<b>Chart Type:</b> {insight.get('chart_type', 'N/A')}<br/>"
            f"<b>Top Results Count:</b> {insight.get('top_n', 'N/A')}",
            body
        ))

        if insight.get("highest_value") is not None:
            story.append(Paragraph(
                f"<b>Highest Value:</b> {insight.get('highest_value')} "
                f"(Count: {insight.get('highest_count')})<br/>"
                f"<b>Lowest Value:</b> {insight.get('lowest_value')} "
                f"(Count: {insight.get('lowest_count')})",
                body
            ))
        else:
            story.append(Paragraph("Selected insight column is not numeric.", note))

        story.append(Paragraph("<b>Insight Result Preview:</b>", body))
        story.append(make_table(insight.get("result"), max_rows=12))
        story.append(Spacer(1, 6))

        insight_fig = report_data.get("insight_figure")
        img = fig_to_rl_image(insight_fig, width_cm=16, height_cm=8.5)
        if img:
            story.append(Paragraph("<b>Insight Chart:</b>", body))
            story.append(img)
        else:
            story.append(Paragraph("Insight chart image skipped because Plotly image export is unavailable on this system.", note))
    else:
        story.append(Paragraph("No insight output generated yet.", note))

    story.append(Spacer(1, 12))

    # Compare section
    story.append(Paragraph("Compare Output", h1))
    compare = report_data.get("compare_output")

    if compare:
        story.append(Paragraph(
            f"<b>Compare Columns:</b> {compare.get('compare_col1', 'N/A')} vs {compare.get('compare_col2', 'N/A')}<br/>"
            f"<b>Compare Chart Type:</b> {compare.get('compare_chart_type', 'N/A')}",
            body
        ))

        story.append(Paragraph("<b>Compare Result Preview:</b>", body))
        story.append(make_table(compare.get("result"), max_rows=12))
        story.append(Spacer(1, 6))

        compare_fig = report_data.get("compare_figure")
        img = fig_to_rl_image(compare_fig, width_cm=16, height_cm=8.5)
        if img:
            story.append(Paragraph("<b>Compare Chart:</b>", body))
            story.append(img)
        else:
            story.append(Paragraph("Compare chart image skipped because Plotly image export is unavailable on this system.", note))
    else:
        story.append(Paragraph("No compare output generated yet.", note))
    smart_insights = report_data.get("smart_insights", [])

    story.append(Spacer(1, 12))

    # Smart Insights section
    story.append(Spacer(1, 12))
    smart_insights = report_data.get("smart_insights", [])

    if smart_insights:
        story.append(Paragraph("Smart Insights", h1))
        story.append(Spacer(1, 8))

        for insight_item in smart_insights:
            story.append(Paragraph(f"• {insight_item}", body))
            story.append(Spacer(1, 4))

        insight_fig = report_data.get("insight_figure")
        if insight_fig is not None:
            img = fig_to_rl_image(insight_fig, width_cm=16, height_cm=8.5)
            if img:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Smart Insights Chart:</b>", body))
                story.append(Spacer(1, 4))
                story.append(img)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(
                    "Smart insights chart image skipped because Plotly image export is unavailable on this system.",
                    note))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No smart insights available.", note))

    # Chat with Data section
    story.append(Spacer(1, 12))
    chat_history = report_data.get("chat_history", [])

    if chat_history:
        story.append(Paragraph("Chat with Data Output", h1))
        story.append(Spacer(1, 8))

        for i, chat in enumerate(chat_history, start=1):
            question = chat.get("question", "")
            text = chat.get("text", "")
            summary = chat.get("summary", "")
            data = chat.get("data", None)
            figure = chat.get("figure", None)

            story.append(Paragraph(f"Q{i}: {question}", h2))
            story.append(Spacer(1, 4))

            if text:
                story.append(Paragraph(f"<b>Answer:</b> {text}", body))
                story.append(Spacer(1, 4))

            if summary:
                story.append(Paragraph(summary, body))
                story.append(Spacer(1, 4))

            if data is not None and not getattr(data, "empty", True):
                story.append(Paragraph("<b>Result Preview:</b>", body))
                story.append(Spacer(1, 4))
                story.append(make_table(data, max_rows=5))
                story.append(Spacer(1, 8))

            if figure is not None:
                img = fig_to_rl_image(figure, width_cm=16, height_cm=8.5)
                if img:
                    story.append(Paragraph("<b>Chart:</b>", body))
                    story.append(Spacer(1, 4))
                    story.append(img)
                    story.append(Spacer(1, 8))
                else:
                    story.append(
                        Paragraph("Chat chart image skipped because Plotly image export is unavailable on this system.",
                                  note))
                    story.append(Spacer(1, 4))

            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No chat history available.", note))

    doc.build(story)
    buffer.seek(0)
    return buffer