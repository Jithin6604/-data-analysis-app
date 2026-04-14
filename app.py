import hashlib
import time
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


from utils.pdf_export import create_full_report_pdf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
INVALID_PLACEHOLDERS = {"UNKNOWN", "ERROR", "N/A", "NA", "-", "--", ""}

DATE_KEYWORDS = {
    "date", "time", "created", "updated", "timestamp",
    "transaction_date", "order_date", "purchase_date", "invoice_date"
}
SALES_KEYWORDS = {
    "sales", "sale", "sales_amount", "total_sales", "revenue",
    "amount", "total_amount", "total_spent", "net_sales", "gross_sales"
}
PROFIT_KEYWORDS = {
    "profit",
    "net_profit",
    "gross_profit",
    "profit_amount",
    "gross_income"   #
}
PRODUCT_KEYWORDS = {"product", "product_name", "item", "item_name", "category", "sku"}
LOCATION_KEYWORDS = {"location", "region", "area", "city", "state", "branch", "place"}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _default_state() -> dict:
    return {
        "chat_history": [],
        "filter_section_ids": [1],
        "show_data_preview": False,
        "insight_output": None,
        "compare_output": None,
        "filter_charts": {},
        "manual_clean_applied": False,
        "cleaning_summary": None,
        "cleaned_df_state": None,
        "uploaded_file_hash": None,
        "dropped_columns": [],
    }


def init_state() -> None:
    defaults = _default_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for sid in st.session_state["filter_section_ids"]:
        if f"filters_{sid}" not in st.session_state:
            st.session_state[f"filters_{sid}"] = [1]
        if f"section_name_{sid}" not in st.session_state:
            st.session_state[f"section_name_{sid}"] = f"Custom View {sid}"


def reset_state(file_hash: str) -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.session_state["uploaded_file_hash"] = file_hash


def get_file_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


# ─────────────────────────────────────────────
# FILTER SECTION MANAGEMENT
# ─────────────────────────────────────────────
def add_filter_section() -> None:
    ids = st.session_state["filter_section_ids"]
    new_id = max(ids) + 1 if ids else 1
    st.session_state["filter_section_ids"].append(new_id)
    st.session_state[f"filters_{new_id}"] = [1]
    st.session_state[f"section_name_{new_id}"] = f"Custom View {new_id}"


def add_filter_row(section_id: int) -> None:
    rows = st.session_state.get(f"filters_{section_id}", [1])
    new_row = max(rows) + 1 if rows else 1
    st.session_state[f"filters_{section_id}"].append(new_row)


def remove_filter_row(section_id: int, filter_id: int) -> None:
    rows = st.session_state.get(f"filters_{section_id}", [1])
    if len(rows) <= 1:
        return

    st.session_state[f"filters_{section_id}"] = [r for r in rows if r != filter_id]

    for key in (
        f"section_{section_id}_filter_col_{filter_id}",
        f"section_{section_id}_filter_val_{filter_id}",
    ):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────
# DATA CLEANING HELPERS
# ─────────────────────────────────────────────
def normalize_col_name(col_name: str) -> str:
    return (
        str(col_name)
        .strip()
        .lower()
        .replace(" ", "_")
    )

def get_safe_selectbox_index(options: list[str], current_value: str, fallback: str = "Auto Detect") -> int:
    if current_value in options:
        return options.index(current_value)
    if fallback in options:
        return options.index(fallback)
    return 0

def normalize_text(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    s = series.astype("string").str.strip()
    s = s.replace(list(INVALID_PLACEHOLDERS), pd.NA)
    s = s.replace({"nan": pd.NA, "None": pd.NA})
    return s


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def drop_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df.columns)
    result = df.loc[:, ~df.columns.duplicated()].copy()
    return result, before - len(result.columns)


def strip_text_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    cleaned = 0
    for col in df.select_dtypes(include=["object", "string"]).columns:
        original = df[col].copy()
        df[col] = df[col].astype("string").str.strip()
        if not original.astype("string").equals(df[col].astype("string")):
            cleaned += 1
    return df, cleaned


def replace_invalid_placeholders(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    total_replaced = 0

    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        before_nulls = s.isna().sum()

        mask_invalid = s.str.upper().isin(INVALID_PLACEHOLDERS - {""}) | s.eq("")
        s = s.mask(mask_invalid, pd.NA)

        total_replaced += max(0, int(s.isna().sum() - before_nulls))
        df[col] = s

    return df, total_replaced


def _looks_numeric(series: pd.Series) -> bool:
    s = normalize_text(series).dropna()
    if s.empty:
        return False
    return pd.to_numeric(s, errors="coerce").notna().mean() >= 0.75


def fix_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    fixed = 0

    for col in df.columns:
        if df[col].dtype in ["object", "string"] and _looks_numeric(df[col]):
            norm = normalize_text(df[col])
            converted = pd.to_numeric(norm, errors="coerce")
            fixed += int((norm.notna() & converted.isna()).sum())
            df[col] = converted

    return df, fixed


def _looks_like_date_column(col_name: str, series: pd.Series) -> bool:
    name_match = any(kw in str(col_name).lower() for kw in DATE_KEYWORDS)
    s = normalize_text(series).dropna()
    if s.empty:
        return False
    parse_ratio = pd.to_datetime(s, errors="coerce").notna().mean()
    return (name_match and parse_ratio >= 0.5) or (parse_ratio >= 0.85)


def fix_date_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    fixed = 0

    for col in df.columns:
        if df[col].dtype in ["object", "string"] and _looks_like_date_column(col, df[col]):
            norm = normalize_text(df[col])
            converted = pd.to_datetime(norm, errors="coerce")
            fixed += int((norm.notna() & converted.isna()).sum())
            df[col] = converted

    return df, fixed


def drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(how="all").copy()
    return df, before - len(df)


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().copy()
    return df, before - len(df)


def remove_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    before = len(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df, 0

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if len(valid) < 8:
            continue

        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        mask = s.isna() | s.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask]

    return df.copy(), before - len(df)


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    *,
    standardize_cols: bool,
    remove_dup_cols: bool,
    strip_spaces: bool,
    fix_placeholders: bool,
    fix_numeric: bool,
    fix_dates: bool,
    remove_empty: bool,
    remove_duplicates: bool,
    remove_outliers: bool,
) -> tuple[pd.DataFrame, dict]:
    wdf = df.copy()
    summary = {
        "columns_standardized": False,
        "duplicate_columns_removed": 0,
        "text_columns_cleaned": 0,
        "placeholder_values_fixed": 0,
        "invalid_numeric_values_fixed": 0,
        "invalid_date_values_fixed": 0,
        "empty_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "outliers_removed": 0,
        "total_rows_removed": 0,
    }

    if standardize_cols:
        wdf = standardize_column_names(wdf)
        summary["columns_standardized"] = True

    if remove_dup_cols:
        wdf, n = drop_duplicate_columns(wdf)
        summary["duplicate_columns_removed"] = n

    if strip_spaces:
        wdf, n = strip_text_whitespace(wdf)
        summary["text_columns_cleaned"] = n

    if fix_placeholders:
        wdf, n = replace_invalid_placeholders(wdf)
        summary["placeholder_values_fixed"] = n

    if fix_numeric:
        wdf, n = fix_numeric_columns(wdf)
        summary["invalid_numeric_values_fixed"] = n

    if fix_dates:
        wdf, n = fix_date_columns(wdf)
        summary["invalid_date_values_fixed"] = n

    if remove_empty:
        wdf, n = drop_empty_rows(wdf)
        summary["empty_rows_removed"] = n

    if remove_duplicates:
        wdf, n = drop_duplicate_rows(wdf)
        summary["duplicate_rows_removed"] = n

    if remove_outliers:
        wdf, n = remove_outliers_iqr(wdf)
        summary["outliers_removed"] = n

    summary["total_rows_removed"] = int(len(df) - len(wdf))
    return wdf, summary


# ─────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────
def _col_name_matches(col: str, keywords: set) -> bool:
    name = str(col).lower()
    return any(kw == name or kw in name for kw in keywords)


def detect_sales_column(df: pd.DataFrame) -> Optional[str]:
    candidates = []

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        name = str(col).lower()
        score = 0

        if _col_name_matches(col, SALES_KEYWORDS):
            score += 10

        for penalised in ("qty", "quantity", "id", "stock", "discount", "tax"):
            if penalised in name:
                score -= 4 if penalised in ("qty", "quantity") else 6 if penalised in ("id", "stock") else 2

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty and s.abs().sum() > 0:
            score += 2

        candidates.append((score, col))

    candidates.sort(reverse=True)
    if candidates and candidates[0][0] >= 4:
        return candidates[0][1]
    return None


def detect_profit_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _col_name_matches(col, PROFIT_KEYWORDS):
            return col
    return None


def detect_product_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _col_name_matches(col, PRODUCT_KEYWORDS):
            return col
    return None


def detect_location_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _col_name_matches(col, LOCATION_KEYWORDS):
            return col
    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    preferred, fallback = [], []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if any(kw in str(col).lower() for kw in DATE_KEYWORDS):
                preferred.append(col)
            else:
                fallback.append(col)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]

    for col in df.columns:
        if _looks_like_date_column(col, df[col]):
            return col

    return None


def profile_dataset(df: pd.DataFrame) -> dict:
    return {
        "sales_col": detect_sales_column(df),
        "product_col": detect_product_column(df),
        "location_col": detect_location_column(df),
        "date_col": detect_date_column(df),
        "profit_col": detect_profit_column(df),
    }


# ─────────────────────────────────────────────
# FILTERING UTILITIES
# ─────────────────────────────────────────────
def get_column_filter_options(series: pd.Series) -> list[str]:
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dropna().dt.strftime("%Y-%m-%d").unique().tolist()
        return sorted(values)

    if pd.api.types.is_numeric_dtype(series):
        try:
            return [str(x) for x in sorted(series.dropna().unique().tolist())]
        except Exception:
            return [str(x) for x in series.dropna().unique().tolist()]

    return sorted(series.dropna().astype(str).unique().tolist())


def apply_column_filter(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    s = df[column]

    if pd.api.types.is_datetime64_any_dtype(s):
        return df[s.dt.strftime("%Y-%m-%d") == value].copy()

    if pd.api.types.is_numeric_dtype(s):
        try:
            return df[pd.to_numeric(s, errors="coerce") == float(value)].copy()
        except Exception:
            return df[s.astype(str) == value].copy()

    return df[s.astype(str) == value].copy()


def get_section_filtered_df(df: pd.DataFrame, section_id: int) -> tuple[pd.DataFrame, list]:
    result = df.copy()
    applied = []

    for fid in st.session_state.get(f"filters_{section_id}", [1]):
        col = st.session_state.get(f"section_{section_id}_filter_col_{fid}", "None")
        val = st.session_state.get(f"section_{section_id}_filter_val_{fid}", None)

        if col and col != "None" and val not in (None, "Select a column first") and col in result.columns:
            result = apply_column_filter(result, col, str(val))
            applied.append((col, val))

    return result, applied


def section_kpis(filtered: pd.DataFrame, total: pd.DataFrame) -> dict:
    row_count = len(filtered)
    total_rows = len(total)
    pct = round(row_count / total_rows * 100, 1) if total_rows else 0

    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    sales_col = sales_col_override

    if sales_col:
        sum_col = sales_col
        sum_val = pd.to_numeric(filtered[sales_col], errors="coerce").sum()
    elif numeric_cols:
        sum_col = numeric_cols[0]
        sum_val = pd.to_numeric(filtered[sum_col], errors="coerce").sum()
    else:
        sum_col = None
        sum_val = None

    return {
        "filtered_rows": row_count,
        "percent_of_total": pct,
        "numeric_col_count": len(numeric_cols),
        "sum_col": sum_col,
        "sum_value": sum_val,
    }


def compare_source_df(final_df: pd.DataFrame) -> pd.DataFrame:
    if not st.session_state.get("use_filtered_data_for_compare"):
        return final_df.copy()

    ids = st.session_state["filter_section_ids"]
    if not ids:
        return final_df.copy()

    filtered, _ = get_section_filtered_df(final_df, ids[0])
    return filtered if not filtered.empty else final_df.copy()


def ensure_compare_col_keys(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    if not cols:
        return

    if "compare_col1" not in st.session_state or st.session_state["compare_col1"] not in cols:
        st.session_state["compare_col1"] = cols[0]

    if "compare_col2" not in st.session_state or st.session_state["compare_col2"] not in cols:
        st.session_state["compare_col2"] = cols[1] if len(cols) > 1 else cols[0]


# ─────────────────────────────────────────────
# SMART INSIGHTS ENGINE
# ─────────────────────────────────────────────
import re

def extract_top_n(query):
    match = re.search(r"\d+", query)
    if match:
        return int(match.group())
    return 5  # default value
def has_any(text: str, words: list[str]) -> bool:
    text = text.lower()
    return any(word in text for word in words)

def ai_understand_question(query: str) -> dict:
    query = query.lower()

    result = {
        "intent": None,
        "top_n": extract_top_n(query),
        "location": None,
        "year": None,
        "month": None
    }

    # detect location
    for loc in dash_df[location_col].dropna().astype(str).unique():
        if loc.lower() in query:
            result["location"] = loc
            break

    # detect year
    for year in ["2020","2021","2022","2023","2024","2025","2026"]:
        if year in query:
            result["year"] = int(year)

    # detect month
    months = {
        "january":1,"february":2,"march":3,"april":4,
        "may":5,"june":6,"july":7,"august":8,
        "september":9,"october":10,"november":11,"december":12
    }

    for m, num in months.items():
        if m in query:
            result["month"] = num

    # detect intent (reuse your logic)
    result["intent"] = detect_chat_intent(query)

    return result


def detect_chat_intent(user_query: str) -> str:
    q = user_query.lower().strip()

    sales_words = ["sales", "revenue", "sale amount", "turnover"]
    profit_words = ["profit", "income", "gross income", "net profit"]
    top_words = ["top", "best", "highest", "most", "leading"]
    worst_words = ["worst", "lowest", "least", "weakest", "underperforming"]
    product_words = ["product", "products", "item", "items", "category", "categories"]
    location_words = ["location", "locations", "city", "cities", "region", "regions", "branch", "branches", "area"]
    trend_words = ["trend", "over time", "monthly", "by month", "growth", "change over time"]
    compare_words = ["compare", "vs", "versus", "difference between"]
    total_words = ["total", "sum", "overall"]
    average_words = ["average", "avg", "mean"]
    row_words = ["rows", "records", "entries"]

    if has_any(q, total_words) and has_any(q, sales_words):
        return "total_sales"

    if has_any(q, total_words) and has_any(q, profit_words):
        return "total_profit"

    if has_any(q, average_words) and has_any(q, sales_words):
        return "average_sales"

    if has_any(q, row_words) or "how many rows" in q or "record count" in q:
        return "total_rows"

    if has_any(q, profit_words) and has_any(q, trend_words):
        return "profit_trend"

    if has_any(q, sales_words) and has_any(q, trend_words):
        return "sales_trend"

    if has_any(q, top_words) and has_any(q, product_words):
        return "top_products"

    if has_any(q, worst_words) and has_any(q, product_words):
        return "worst_products"

    if has_any(q, top_words) and has_any(q, location_words):
        return "top_locations"

    if has_any(q, worst_words) and has_any(q, location_words):
        return "worst_locations"

    if "sales by" in q and has_any(q, location_words):
        return "sales_by"

    if "profit by" in q and has_any(q, product_words):
        return "profit_by"

    if has_any(q, compare_words):
        return "compare_products"

    return "unknown"

def generate_smart_insights(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["No data available for insights."]

    insights = []
    wdf = df.copy()
    sales_col = sales_col_override
    product_col = product_col_override
    location_col = location_col_override
    date_col = date_col_override

    if sales_col:
        wdf[sales_col] = pd.to_numeric(wdf[sales_col], errors="coerce")

    if product_col and sales_col:
        temp = wdf[[product_col, sales_col]].dropna()
        if not temp.empty:
            by_product = (
                temp.groupby(product_col, as_index=False)[sales_col]
                .sum()
                .sort_values(sales_col, ascending=False)
            )

            if not by_product.empty:
                insights.append(f"Top-selling product is {by_product.iloc[0][product_col]}.")

                avg = by_product[sales_col].mean()
                weakest = by_product.iloc[-1]

                if weakest[sales_col] == 0:
                    insights.append(f"{weakest[product_col]} has almost no sales.")
                elif weakest[sales_col] < 0.5 * avg:
                    insights.append(f"{weakest[product_col]} is underperforming compared to other products.")

                total = by_product[sales_col].sum()
                if total > 0:
                    top3_pct = by_product[sales_col].head(3).sum() / total * 100
                    insights.append(f"Top 3 products contribute {top3_pct:.1f}% of total sales.")

    if product_col and location_col and sales_col:
        temp = wdf[[product_col, location_col, sales_col]].dropna()
        if not temp.empty:
            grouped = temp.groupby([product_col, location_col], as_index=False)[sales_col].sum()
            shown = 0

            for product in grouped[product_col].dropna().astype(str).unique():
                subset = grouped[grouped[product_col].astype(str) == product]
                if not subset.empty:
                    top_area = subset.sort_values(sales_col, ascending=False).iloc[0]
                    insights.append(f"{top_area[location_col]} is the strongest area for {product}.")
                    shown += 1
                if shown >= 2:
                    break

    if location_col and sales_col:
        temp = wdf[[location_col, sales_col]].dropna()
        if not temp.empty:
            by_loc = temp.groupby(location_col, as_index=False)[sales_col].sum()
            avg = by_loc[sales_col].mean()
            weakest = by_loc.sort_values(sales_col).iloc[0]

            if weakest[sales_col] < 0.5 * avg:
                insights.append(f"{weakest[location_col]} is underperforming compared to other locations.")

    if date_col and sales_col:
        try:
            temp = wdf[[date_col, sales_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()

            if not temp.empty:
                trend = (
                    temp.assign(period=temp[date_col].dt.to_period("M").astype(str))
                    .groupby("period", as_index=False)[sales_col]
                    .sum()
                    .sort_values("period")
                )

                if len(trend) >= 2:
                    last, prev = trend[sales_col].iloc[-1], trend[sales_col].iloc[-2]
                    if prev != 0:
                        pct = (last - prev) / prev * 100
                        if pct < 0:
                            insights.append(f"Sales dropped {abs(pct):.1f}% compared to the last period.")
                        elif pct > 0:
                            insights.append(f"Sales increased {pct:.1f}% compared to the last period.")
                        else:
                            insights.append("Sales stayed flat compared to the last period.")
        except Exception:
            pass

    if not insights:
        insights.append("No strong insights found from the current cleaned data.")

    seen = set()
    return [x for x in insights if not (x in seen or seen.add(x))]

def generate_smart_suggestions(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["Upload data to get suggestions."]

    suggestions = []

    sales_col = sales_col_override
    profit_col = profit_col_override
    product_col = product_col_override
    location_col = location_col_override
    date_col = date_col_override

    if sales_col:
        suggestions.append("Show total sales")
        suggestions.append("Show average sales")

    if product_col and sales_col:
        suggestions.append("Show top 5 products")
        suggestions.append("Show worst 5 products")

    if location_col and sales_col:
        suggestions.append("Show top locations")
        suggestions.append("Show worst locations")

    if date_col and sales_col:
        suggestions.append("Show sales trend")

    if date_col and profit_col:
        suggestions.append("Show profit trend")

    if product_col and profit_col:
        suggestions.append("Show profit by product")

    if product_col and product_col in df.columns:
        unique_products = df[product_col].dropna().astype(str).unique().tolist()
        if len(unique_products) >= 2:
            suggestions.append(f"Compare {unique_products[0]} and {unique_products[1]}")

    seen = set()
    clean_suggestions = []
    for s in suggestions:
        if s not in seen:
            clean_suggestions.append(s)
            seen.add(s)

    return clean_suggestions[:6]


# ─────────────────────────────────────────────
# SCROLL HELPER
# ─────────────────────────────────────────────
def scroll_to(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
            const el = window.parent.document.getElementById("{anchor_id}");
            if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
        </script>
        """,
        height=0,
    )


# ─────────────────────────────────────────────
# FIGURE ACCESSORS
# ─────────────────────────────────────────────
def _get_figure(state_key: str) -> Optional[Any]:
    data = st.session_state.get(state_key)
    return data.get("figure") if data else None


def get_insight_figure():
    return _get_figure("insight_output")


def get_compare_figure():
    return _get_figure("compare_output")


def get_section_figure(section_id: int) -> Optional[Any]:
    charts = st.session_state.get("filter_charts", {})
    entry = charts.get(section_id)
    return entry.get("figure") if entry else None


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
.chat-shell {
    border: 1px solid #e5ecf6;
    border-radius: 20px;
    padding: 18px 18px 14px 18px;
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    box-shadow: 0 8px 24px rgba(16, 24, 40, 0.04);
    margin-top: 12px;
    margin-bottom: 18px;
}

.chat-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 2px;
}

.small-muted {
    color: #6b7280;
    font-size: 0.92rem;
    margin-bottom: 0.4rem;
}

.divider-soft {
    display: none;
}

[data-testid="stTextInput"] > div > div {
    border-radius: 14px !important;
}

[data-testid="stTextInput"] input {
    border-radius: 14px !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
}

.chat-input-label {
    font-size: 0.92rem;
    font-weight: 600;
    color: #374151;
    margin: 10px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 1.2rem;
    max-width: 1450px;
}

.main-title {
    font-size: 2.05rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
    letter-spacing: -0.3px;
}

.sub-title {
    color: #6b7280;
    margin-bottom: 0.8rem;
    font-size: 0.96rem;
}

.section-card {
    background: #f8fbff;
    border: 1px solid #dbe7f3;
    border-radius: 18px;
    padding: 16px 16px 14px 16px;
    box-shadow: none;
    margin-bottom: 14px;
}

.kpi-card {
    background: #eef5ff;
    border: 1px solid #d7e6fb;
    border-radius: 16px;
    padding: 10px 12px 2px 12px;
    box-shadow: none;
}

.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
    margin-bottom: 0.2rem;
}

.chat-shell {
    border: 1px solid #dbe7f3;
    border-radius: 18px;
    padding: 14px;
    background: #f8fbff;
    box-shadow: none;
    margin-top: 8px;
    margin-bottom: 14px;
}

.chat-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 2px;
}

.divider-soft {
    display: none;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 6px 12px !important;
}

[data-testid="stMetric"] {
    background: transparent;
    border-radius: 12px;
    padding: 2px 0 0 0;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}

[data-testid="stTextInput"] > div > div {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────
init_state()

st.markdown('<div class="main-title">AI Data Analysis Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload business data, clean it manually, create custom views, '
    'analyze patterns, compare results, generate smart insights, and download a full PDF report.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Control Panel")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.subheader("Navigation")
    nav_choice = st.radio(
        "Go to",
        ["Overview", "Cleaning", "Dashboard", "Custom Views", "Insights", "Compare", "Download"],
        label_visibility="collapsed",
    )

if not uploaded_file:
    st.info("Upload a CSV file from the sidebar to start.")
    st.stop()

current_hash = get_file_hash(uploaded_file)
if st.session_state.get("uploaded_file_hash") != current_hash:
    reset_state(current_hash)

# ─────────────────────────────────────────────
# LOAD RAW DATA
# ─────────────────────────────────────────────
raw_df = pd.read_csv(uploaded_file)
raw_df.columns = raw_df.columns.astype(str).str.strip()
original_row_count = len(raw_df)

# ─────────────────────────────────────────────
# NAVIGATION SCROLL
# ─────────────────────────────────────────────
NAV_ANCHORS = {
    "Overview": "overview_section",
    "Cleaning": "cleaning_section",
    "Dashboard": "dashboard_section",
    "Custom Views": "custom_views_section",
    "Insights": "insights_section",
    "Compare": "compare_section",
    "Download": "download_section",
}
scroll_to(NAV_ANCHORS[nav_choice])

nav_cols = st.columns(7)
for widget, (label, anchor) in zip(nav_cols, NAV_ANCHORS.items()):
    short = label.split()[0]
    with widget:
        if st.button(short):
            scroll_to(anchor)

# ─────────────────────────────────────────────
# HELPERS FOR SAFE COLUMN SELECTION
# ─────────────────────────────────────────────
def normalize_col_name(col_name: str) -> str:
    return (
        str(col_name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def get_safe_selectbox_index(options: list[str], current_value: str, fallback: str = "Auto Detect") -> int:
    if current_value in options:
        return options.index(current_value)
    if fallback in options:
        return options.index(fallback)
    return 0


# ─────────────────────────────────────────────
# BUILD WORKING DATAFRAMES
# ─────────────────────────────────────────────
if st.session_state.get("manual_clean_applied") and st.session_state.get("cleaned_df_state") is not None:
    cleaned_base_df = st.session_state["cleaned_df_state"].copy()
else:
    cleaned_base_df = raw_df.copy()

dropped_columns = st.session_state.get("dropped_columns", [])
final_df = cleaned_base_df.drop(
    columns=[c for c in dropped_columns if c in cleaned_base_df.columns]
).copy()

# NOW final_df exists, so profile it here
data_profile = profile_dataset(final_df)

detected_sales = data_profile["sales_col"] or None
detected_profit = data_profile["profit_col"] or None
detected_product = data_profile["product_col"] or None
detected_location = data_profile["location_col"] or None
detected_date = data_profile["date_col"] or None

all_column_options = ["None", "Auto Detect"] + list(final_df.columns)

# previous selections from session state
prev_sales = st.session_state.get("manual_sales_col", "Auto Detect")
prev_profit = st.session_state.get("manual_profit_col", "Auto Detect")
prev_product = st.session_state.get("manual_product_col", "Auto Detect")
prev_location = st.session_state.get("manual_location_col", "Auto Detect")
prev_date = st.session_state.get("manual_date_col", "Auto Detect")

# remap old names to cleaned names if columns were standardized
normalized_map = {normalize_col_name(col): col for col in final_df.columns}

def remap_selection(value: str) -> str:
    if value in ("None", "Auto Detect"):
        return value
    if value in final_df.columns:
        return value
    normalized_value = normalize_col_name(value)
    return normalized_map.get(normalized_value, "Auto Detect")

prev_sales = remap_selection(prev_sales)
prev_profit = remap_selection(prev_profit)
prev_product = remap_selection(prev_product)
prev_location = remap_selection(prev_location)
prev_date = remap_selection(prev_date)

col1, col2 = st.columns(2)

with col1:
    selected_sales_col = st.selectbox(
        "Select Sales Column",
        all_column_options,
        index=get_safe_selectbox_index(all_column_options, prev_sales),
        key="manual_sales_col"
    )

with col2:
    selected_profit_col = st.selectbox(
        "Select Profit Column",
        all_column_options,
        index=get_safe_selectbox_index(all_column_options, prev_profit),
        key="manual_profit_col"
    )

col3, col4, col5 = st.columns(3)

with col3:
    selected_product_col = st.selectbox(
        "Select Product Column",
        all_column_options,
        index=get_safe_selectbox_index(all_column_options, prev_product),
        key="manual_product_col"
    )

with col4:
    selected_location_col = st.selectbox(
        "Select Location Column",
        all_column_options,
        index=get_safe_selectbox_index(all_column_options, prev_location),
        key="manual_location_col"
    )

with col5:
    selected_date_col = st.selectbox(
        "Select Date Column",
        all_column_options,
        index=get_safe_selectbox_index(all_column_options, prev_date),
        key="manual_date_col"
    )

sales_col_override = (
    None if selected_sales_col == "None"
    else detected_sales if selected_sales_col == "Auto Detect"
    else selected_sales_col
)

profit_col_override = (
    None if selected_profit_col == "None"
    else detected_profit if selected_profit_col == "Auto Detect"
    else selected_profit_col
)

product_col_override = (
    None if selected_product_col == "None"
    else detected_product if selected_product_col == "Auto Detect"
    else selected_product_col
)

location_col_override = (
    None if selected_location_col == "None"
    else detected_location if selected_location_col == "Auto Detect"
    else selected_location_col
)

date_col_override = (
    None if selected_date_col == "None"
    else detected_date if selected_date_col == "Auto Detect"
    else selected_date_col
)

st.caption(f"Using Sales Column: {sales_col_override if sales_col_override else 'None'}")
st.caption(f"Using Profit Column: {profit_col_override if profit_col_override else 'None'}")
st.caption(f"Using Product Column: {product_col_override if product_col_override else 'None'}")
st.caption(f"Using Location Column: {location_col_override if location_col_override else 'None'}")
st.caption(f"Using Date Column: {date_col_override if date_col_override else 'None'}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="overview_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Overview")

current_row_count = len(final_df)
rows_removed = original_row_count - current_row_count
dropped_col_count = len(dropped_columns)
m1, m2, m3, m4 = st.columns(4)

for widget, label, value in zip(
    (m1, m2, m3, m4),
    ("Original Rows", "Current Rows", "Removed Rows", "Dropped Columns"),
    (original_row_count, current_row_count, rows_removed, dropped_col_count),
):
    with widget:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(label, value)
        st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="cleaning_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Cleaning")

# 🔄 Reset Cleaning button INSIDE cleaning section
if st.button("🔄 Reset Cleaning", key="reset_cleaning_btn"):
    st.session_state["manual_clean_applied"] = False
    st.session_state["cleaned_df_state"] = None
    st.session_state["cleaning_summary"] = None
    st.rerun()

pre_clean_df = raw_df.copy()

cols_to_drop_pre = st.multiselect(
    "Select columns to remove",
    pre_clean_df.columns.tolist(),
    default=st.session_state.get("dropped_columns", []),
    key="columns_to_drop_pre_clean",
)

st.session_state["dropped_columns"] = cols_to_drop_pre

if cols_to_drop_pre:
    pre_clean_df = pre_clean_df.drop(columns=cols_to_drop_pre)
    st.success("Dropped columns: " + ", ".join(cols_to_drop_pre))
else:
    st.info("No columns dropped")

st.write("### Manual Cleaning Options")
opt1, opt2, opt3 = st.columns(3)

with opt1:
    do_standardize_cols = st.checkbox("Standardize column names", value=True)
    do_remove_dup_cols = st.checkbox("Remove duplicate columns", value=True)
    do_strip_spaces = st.checkbox("Clean text spacing", value=True)

with opt2:
    do_fix_placeholders = st.checkbox("Fix UNKNOWN / ERROR values", value=True)
    do_fix_numeric = st.checkbox("Convert bad numeric values", value=True)
    do_fix_dates = st.checkbox("Convert bad date values", value=True)

with opt3:
    do_remove_empty = st.checkbox("Remove empty rows", value=True)
    do_remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    do_remove_outliers = st.checkbox("Filter outliers", value=False)

if st.button("Apply Manual Cleaning", use_container_width=True):
    result_df, result_summary = apply_cleaning_pipeline(
        pre_clean_df,
        standardize_cols=do_standardize_cols,
        remove_dup_cols=do_remove_dup_cols,
        strip_spaces=do_strip_spaces,
        fix_placeholders=do_fix_placeholders,
        fix_numeric=do_fix_numeric,
        fix_dates=do_fix_dates,
        remove_empty=do_remove_empty,
        remove_duplicates=do_remove_duplicates,
        remove_outliers=do_remove_outliers,
    )

    st.session_state["manual_clean_applied"] = True
    st.session_state["cleaning_summary"] = result_summary
    st.session_state["cleaned_df_state"] = result_df
    st.rerun()

if st.session_state.get("manual_clean_applied") and st.session_state.get("cleaned_df_state") is not None:
    cleaning_df = st.session_state["cleaned_df_state"].copy()
else:
    cleaning_df = pre_clean_df.copy()

clean_sales_col = detect_sales_column(cleaning_df)
clean_location_col = detect_location_column(cleaning_df)

avg_order_value = None
if clean_sales_col and clean_sales_col in cleaning_df.columns:
    vals = pd.to_numeric(cleaning_df[clean_sales_col], errors="coerce").dropna()
    if not vals.empty:
        avg_order_value = vals.mean()

unique_locations = None
if clean_location_col and clean_location_col in cleaning_df.columns:
    unique_locations = cleaning_df[clean_location_col].dropna().astype(str).nunique()

kpi_labels = [
    "Original Rows", "Current Rows", "Removed Rows",
    "Dropped Columns", "Avg Order Value", "Outlets / Locations"
]
kpi_values = [
    original_row_count,
    len(cleaning_df),
    original_row_count - len(cleaning_df),
    len(cols_to_drop_pre),
    round(avg_order_value, 2) if avg_order_value is not None else "N/A",
    unique_locations if unique_locations is not None else "N/A",
]

kpi_cols = st.columns(6)
for widget, label, value in zip(kpi_cols, kpi_labels, kpi_values):
    with widget:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(label, value)
        st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("manual_clean_applied") and st.session_state.get("cleaning_summary"):
    s = st.session_state["cleaning_summary"]
    st.success("Cleaning applied successfully")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Empty Rows Removed", s["empty_rows_removed"])
    with s2:
        st.metric("Duplicate Rows Removed", s["duplicate_rows_removed"])
    with s3:
        st.metric("Placeholder Values Fixed", s["placeholder_values_fixed"])
    with s4:
        st.metric("Outliers Removed", s["outliers_removed"])

    st.write("### Cleaning Summary")
    lines = []

    if cols_to_drop_pre:
        lines.append(f"Dropped columns: {', '.join(cols_to_drop_pre)}")
    if s["columns_standardized"]:
        lines.append("Standardized column names")
    if s["duplicate_columns_removed"]:
        lines.append(f"Removed {s['duplicate_columns_removed']} duplicate columns")
    if s["text_columns_cleaned"]:
        lines.append(f"Cleaned spacing in {s['text_columns_cleaned']} text columns")
    if s["placeholder_values_fixed"]:
        lines.append(f"Replaced {s['placeholder_values_fixed']} invalid placeholders")
    if s["invalid_numeric_values_fixed"]:
        lines.append(f"Converted {s['invalid_numeric_values_fixed']} invalid numeric values")
    if s["invalid_date_values_fixed"]:
        lines.append(f"Converted {s['invalid_date_values_fixed']} invalid date values")
    if s["empty_rows_removed"]:
        lines.append(f"Removed {s['empty_rows_removed']} empty rows")
    if s["duplicate_rows_removed"]:
        lines.append(f"Removed {s['duplicate_rows_removed']} duplicate rows")
    if s["outliers_removed"]:
        lines.append(f"Removed {s['outliers_removed']} outlier rows")
    if s["total_rows_removed"]:
        lines.append(f"Total rows removed from original data: {s['total_rows_removed']}")
    if not lines:
        lines.append("No cleaning changes were needed.")

    for line in lines:
        st.write(f"• {line}")

# ✅ Added final trust summary INSIDE cleaning section
st.markdown("### 📊 Final Cleaning Summary")

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
sum_col1.metric("Original Rows", original_row_count)
sum_col2.metric("Final Rows", len(cleaning_df))
sum_col3.metric("Rows Removed", original_row_count - len(cleaning_df))
sum_col4.metric("Columns Dropped", len(cols_to_drop_pre))

missing_before = raw_df.isnull().sum().sum()
missing_after = cleaning_df.isnull().sum().sum()

st.write(f"✔ Missing before: {missing_before}")
st.write(f"✔ Missing after: {missing_after}")
st.info("✅ Analysis is based on cleaned dataset")

left_col, right_col = st.columns(2)

with left_col:
    st.write("**Current Columns**")
    st.write(list(cleaning_df.columns))

    st.download_button(
        "Download Cleaned CSV",
        data=cleaning_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

with right_col:
    st.write("**Cleaned Data Preview**")
    st.dataframe(cleaning_df.head(20), use_container_width=True)

if st.button("Show / Hide Raw vs Current Preview"):
    st.session_state["show_data_preview"] = not st.session_state["show_data_preview"]

if st.session_state["show_data_preview"]:
    p1, p2 = st.columns(2)
    with p1:
        st.write("**Raw Data**")
        st.dataframe(raw_df.head(20), use_container_width=True)
    with p2:
        st.write("**Current Cleaned Data**")
        st.dataframe(cleaning_df.head(20), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="dashboard_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Dashboard")

if not st.session_state.get("manual_clean_applied"):
    st.warning("Please clean your data first to unlock dashboard analysis.")
else:
    dash_df = final_df.copy()
    p = data_profile

    product_col = product_col_override if product_col_override else p["product_col"]
    location_col = location_col_override if location_col_override else p["location_col"]
    sales_col = sales_col_override if sales_col_override else p["sales_col"]
    date_col = date_col_override if date_col_override else p["date_col"]
    profit_col = profit_col_override if profit_col_override else p["profit_col"]

    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        if product_col:
            options = ["All"] + get_column_filter_options(dash_df[product_col])
            sel = st.selectbox("Product Filter", options, key="dash_product_filter")
            if sel != "All":
                dash_df = apply_column_filter(dash_df, product_col, sel)
        else:
            st.selectbox("Product Filter", ["Not available"], disabled=True)

    with fc2:
        if location_col:
            options = ["All"] + get_column_filter_options(dash_df[location_col])
            sel = st.selectbox("Location Filter", options, key="dash_location_filter")
            if sel != "All":
                dash_df = apply_column_filter(dash_df, location_col, sel)
        else:
            st.selectbox("Location Filter", ["Not available"], disabled=True)

    with fc3:
        if date_col and pd.api.types.is_datetime64_any_dtype(dash_df[date_col]):
            dates = dash_df[date_col].dropna()
            if not dates.empty:
                date_range = st.date_input(
                    "Date Range",
                    value=(dates.min().date(), dates.max().date()),
                    key="dash_date_range",
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = date_range
                    dash_df = dash_df[dash_df[date_col].dt.date.between(start, end)].copy()
            else:
                st.selectbox("Date Filter", ["Not available"], disabled=True)
        else:
            st.selectbox("Date Filter", ["Not available"], disabled=True)

    with fc4:
        st.metric("Filtered Rows", len(dash_df))

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("Total Rows", len(dash_df))

    with k2:
        if sales_col:
            sv = pd.to_numeric(dash_df[sales_col], errors="coerce").dropna()
            st.metric("Total Sales", round(sv.sum(), 2) if not sv.empty else "N/A")
        else:
            st.metric("Total Sales", "N/A")

    with k3:
        if profit_col:
            pv = pd.to_numeric(dash_df[profit_col], errors="coerce").dropna()
            st.metric("Total Profit", round(pv.sum(), 2) if not pv.empty else "N/A")
        else:
            st.metric("Total Profit", "N/A")

    with k4:
        if sales_col:
            sv = pd.to_numeric(dash_df[sales_col], errors="coerce").dropna()
            st.metric("Avg Order Value", round(sv.mean(), 2) if not sv.empty else "N/A")
        else:
            st.metric("Avg Order Value", "N/A")

    with k5:
        if product_col and sales_col:
            temp = dash_df[[product_col, sales_col]].copy()
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                top = temp.groupby(product_col)[sales_col].sum().idxmax()
                st.metric("Top Product", str(top))
            else:
                st.metric("Top Product", "N/A")
        else:
            st.metric("Top Product", "N/A")

    with k6:
        if location_col and sales_col:
            temp = dash_df[[location_col, sales_col]].copy()
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                top = temp.groupby(location_col)[sales_col].sum().idxmax()
                st.metric("Top Location", str(top))
            else:
                st.metric("Top Location", "N/A")
        else:
            st.metric("Top Location", "N/A")

    st.write("### 🧠 Smart Insights")
    for insight in generate_smart_insights(dash_df)[:6]:
        st.info(insight)

    cr1a, cr1b = st.columns(2)

    with cr1a:
        st.markdown('<div class="chart-title">Sales by Product</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtle">See which products contribute the most to sales.</div>',
                    unsafe_allow_html=True)
        if product_col and sales_col:
            temp = dash_df[[product_col, sales_col]].copy()
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                grouped = (
                    temp.groupby(product_col, as_index=False)[sales_col]
                    .sum()
                    .sort_values(sales_col, ascending=False)
                    .head(10)
                )
                st.plotly_chart(
                    px.bar(grouped, x=product_col, y=sales_col, title="Sales by Product"),
                    use_container_width=True,
                )
            else:
                st.info("No valid data for this chart.")
        else:
            st.info("Product or sales column not detected.")

    with cr1b:
        st.write("**Sales by Location**")
        if location_col and sales_col:
            temp = dash_df[[location_col, sales_col]].copy()
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                grouped = (
                    temp.groupby(location_col, as_index=False)[sales_col]
                    .sum()
                    .sort_values(sales_col, ascending=False)
                    .head(10)
                )
                st.plotly_chart(
                    px.bar(grouped, x=location_col, y=sales_col, title="Sales by Location"),
                    use_container_width=True,
                )
            else:
                st.info("No valid data for this chart.")
        else:
            st.info("Location or sales column not detected.")

    cr2a, cr2b = st.columns(2)

    with cr2a:
        st.write("**Sales Trend**")
        if date_col and sales_col and pd.api.types.is_datetime64_any_dtype(dash_df[date_col]):
            temp = dash_df[[date_col, sales_col]].copy()
            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                grouped = (
                    temp.assign(period=temp[date_col].dt.to_period("M").astype(str))
                    .groupby("period", as_index=False)[sales_col]
                    .sum()
                    .sort_values("period")
                )
                st.plotly_chart(
                    px.line(grouped, x="period", y=sales_col, markers=True, title="Sales Trend"),
                    use_container_width=True,
                )
            else:
                st.info("No valid date/sales data for trend.")
        else:
            st.info("Date or sales column not detected.")

    with cr2b:
        st.write("**Profit by Product**")
        if product_col and profit_col:
            temp = dash_df[[product_col, profit_col]].copy()
            temp[profit_col] = pd.to_numeric(temp[profit_col], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                grouped = (
                    temp.groupby(product_col, as_index=False)[profit_col]
                    .sum()
                    .sort_values(profit_col, ascending=False)
                    .head(10)
                )
                st.plotly_chart(
                    px.bar(grouped, x=product_col, y=profit_col, title="Profit by Product"),
                    use_container_width=True,
                )
            else:
                st.info("No valid profit data for this chart.")
        else:
            st.info("Product or profit column not detected.")

    chat_title_col, chat_clear_col = st.columns([6, 1])

    with chat_title_col:


        head_left, head_right = st.columns([6, 1])

        # ─────────────────────────────────────────────
        # CHAT WITH DATA
        # ─────────────────────────────────────────────
        st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

        head_left, head_right = st.columns([6, 1])

        with head_left:
            st.markdown('<div class="chat-title">💬 Chat with Your Data</div>', unsafe_allow_html=True)
            st.caption("Powered by your cleaned dataset and intelligent query system")

            st.markdown(
                '<div class="small-muted">Ask things like total sales, top 3 products, top products in Kochi, or show sales trend.</div>',
                unsafe_allow_html=True
            )

        with head_right:
            if st.button("Clear Chat", key="clear_chat_top"):
                st.session_state["chat_history"] = []
                st.rerun()

        # pending input support
        if "pending_chat_input" not in st.session_state:
            st.session_state["pending_chat_input"] = ""

        if "chat_input" not in st.session_state:
            st.session_state["chat_input"] = ""

        if st.session_state["pending_chat_input"]:
            st.session_state["chat_input"] = st.session_state["pending_chat_input"]
            st.session_state["pending_chat_input"] = ""

        # suggested questions stay on top
        st.caption("💡 Suggested questions based on your current cleaned data")

        suggested_questions = generate_smart_suggestions(dash_df)
        suggestion_cols = st.columns(2)

        for idx, suggestion in enumerate(suggested_questions):
            with suggestion_cols[idx % 2]:
                if st.button(suggestion, key=f"suggestion_{idx}"):
                    st.session_state["pending_chat_input"] = suggestion
                    st.rerun()

        # input box stays on top
        st.markdown('<div class="chat-input-label">Ask a question about your data</div>', unsafe_allow_html=True)

        user_query = st.text_input(
            label="",
            placeholder="Example: top 5 products in Kochi",
            key="chat_input"
        )

        # old chat history stays below input
        for i, chat in enumerate(st.session_state["chat_history"]):
            with st.chat_message("user"):
                st.write(chat["question"])

            with st.chat_message("assistant"):
                if chat["text"]:
                    st.success(chat["text"])

                if chat["summary"]:
                    st.info(chat["summary"])

                if chat["data"] is not None or chat["figure"] is not None:
                    left_hist, right_hist = st.columns([1, 2])

                    with left_hist:
                        if chat["data"] is not None:
                            st.dataframe(chat["data"], use_container_width=True)

                    with right_hist:
                        if chat["figure"] is not None:
                            st.plotly_chart(
                                chat["figure"],
                                use_container_width=True,
                                key=f"chat_history_fig_{i}"
                            )

        if user_query:
            with st.spinner("🔍 Analyzing your data and generating insights..."):
                import time

                time.sleep(0.5)

                query = user_query.lower().strip()
                ai_result = ai_understand_question(user_query)

                intent = ai_result["intent"]
                top_n = ai_result["top_n"]
                selected_location = ai_result["location"]
                selected_year = ai_result["year"]
                selected_month = ai_result["month"]
                chat_df = dash_df.copy()

                result_df = None
                result_text = None
                result_fig = None
                result_summary = None

                # TOTAL SALES
                if intent == "total_sales":
                    if sales_col:
                        total = pd.to_numeric(chat_df[sales_col], errors="coerce").sum()
                        total = round(total, 2)

                        result_text = f"Total Sales: {total}"
                        result_summary = f"The total sales for the current data selection is {total}."
                        result_df = pd.DataFrame({
                            "Metric": ["Total Sales"],
                            "Value": [total]
                        })
                        result_fig = px.bar(result_df, x="Metric", y="Value", title="Total Sales")
                    else:
                        result_text = "Sales column not found."

                # TOTAL PROFIT
                elif intent == "total_profit":
                    if profit_col:
                        total = pd.to_numeric(chat_df[profit_col], errors="coerce").sum()
                        total = round(total, 2)

                        result_text = f"Total Profit: {total}"
                        result_summary = f"The total profit for the current data selection is {total}."
                        result_df = pd.DataFrame({
                            "Metric": ["Total Profit"],
                            "Value": [total]
                        })
                        result_fig = px.bar(result_df, x="Metric", y="Value", title="Total Profit")
                    else:
                        result_text = "Profit column not found."

                # AVERAGE SALES
                elif intent == "average_sales":
                    if sales_col:
                        avg = pd.to_numeric(chat_df[sales_col], errors="coerce").mean()
                        avg = round(avg, 2)

                        result_text = f"Average Sales: {avg}"
                        result_summary = f"The average sales value is {avg}."
                        result_df = pd.DataFrame({
                            "Metric": ["Average Sales"],
                            "Value": [avg]
                        })
                        result_fig = px.bar(result_df, x="Metric", y="Value", title="Average Sales")
                    else:
                        result_text = "Sales column not found."

                # TOTAL ROWS
                elif intent == "total_rows":
                    total_rows = len(chat_df)
                    result_text = f"Total Rows: {total_rows}"
                    result_summary = f"There are {total_rows} rows in the current data selection."
                    result_df = pd.DataFrame({
                        "Metric": ["Total Rows"],
                        "Value": [total_rows]
                    })
                    result_fig = px.bar(result_df, x="Metric", y="Value", title="Total Rows")

                # TOP PRODUCTS
                elif intent == "top_products":
                    if product_col and sales_col:
                        top_n = extract_top_n(user_query)
                        temp = chat_df.copy()

                        if location_col and location_col in temp.columns:
                            locations = temp[location_col].dropna().astype(str).unique()
                            for loc in locations:
                                if selected_location and location_col:
                                    temp = temp[temp[location_col].astype(str).str.lower() == selected_location.lower()]
                                    result_text = f"Showing results for location: {loc}"
                                    break

                        if selected_year and date_col:
                            temp = temp[temp[date_col].dt.year == selected_year]

                            for year in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]:
                                if year in query:
                                    temp = temp[temp[date_col].dt.year == int(year)]
                                    result_text = f"{result_text} | Year: {year}" if result_text else f"Showing results for year: {year}"
                                    break

                            month_map = {
                                "january": 1, "february": 2, "march": 3, "april": 4,
                                "may": 5, "june": 6, "july": 7, "august": 8,
                                "september": 9, "october": 10, "november": 11, "december": 12
                            }

                            for month_name, month_num in month_map.items():
                                if selected_month and date_col:
                                    temp = temp[temp[date_col].dt.month == selected_month]
                                    result_text = f"{result_text} | Month: {month_name.title()}" if result_text else f"Showing results for month: {month_name.title()}"
                                    break

                        temp = temp[[product_col, sales_col]].copy()
                        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                        temp = temp.dropna()

                        if not temp.empty:
                            grouped = (
                                temp.groupby(product_col, as_index=False)[sales_col]
                                .sum()
                                .sort_values(sales_col, ascending=False)
                                .head(top_n)
                            )

                            result_df = grouped
                            if not result_text:
                                result_text = f"Top {top_n} Products"

                            top_names = grouped[product_col].astype(str).tolist()
                            if len(top_names) == 1:
                                result_summary = f"The top product is {top_names[0]}."
                            else:
                                result_summary = f"The top {len(top_names)} products are " + ", ".join(
                                    top_names[:-1]) + f", and {top_names[-1]}."

                            result_fig = px.bar(
                                grouped,
                                x=product_col,
                                y=sales_col,
                                title=f"Top {top_n} Products"
                            )
                        else:
                            result_text = "No data found for your query. Try adjusting filters or asking differently."
                    else:
                        result_text = "Product column or Sales column not found."

                # SALES TREND
                elif intent == "sales_trend":
                    if date_col and sales_col:
                        if date_col == sales_col:
                            result_text = "Date column and Sales column cannot be the same."
                        else:
                            temp = chat_df.loc[:, [date_col, sales_col]].copy()
                            temp = temp.loc[:, ~temp.columns.duplicated()]
                            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                            temp = temp.dropna()

                            if not temp.empty:
                                grouped = (
                                    temp.assign(period=temp[date_col].dt.to_period("M").astype(str))
                                    .groupby("period", as_index=False)[sales_col]
                                    .sum()
                                    .sort_values("period")
                                )

                                result_df = grouped
                                result_text = "Sales Trend"
                                result_summary = "This chart shows how sales changed over time by month."
                                result_fig = px.line(
                                    grouped,
                                    x="period",
                                    y=sales_col,
                                    markers=True,
                                    title="Sales Trend"
                                )
                            else:
                                result_text = "No valid date and sales data available for trend."
                    else:
                        result_text = "Date column or Sales column not found."

                # TOP LOCATIONS
                elif intent == "top_locations":
                    if location_col and sales_col:
                        top_n = extract_top_n(user_query)
                        temp = chat_df[[location_col, sales_col]].copy()
                        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                        temp = temp.dropna()

                        if not temp.empty:
                            grouped = (
                                temp.groupby(location_col, as_index=False)[sales_col]
                                .sum()
                                .sort_values(sales_col, ascending=False)
                                .head(top_n)
                            )

                            result_df = grouped
                            result_text = f"Top {top_n} Locations"
                            top_names = grouped[location_col].astype(str).tolist()
                            if len(top_names) == 1:
                                result_summary = f"The top location is {top_names[0]}."
                            else:
                                result_summary = f"The top {len(top_names)} locations are " + ", ".join(
                                    top_names[:-1]) + f", and {top_names[-1]}."

                            result_fig = px.bar(
                                grouped,
                                x=location_col,
                                y=sales_col,
                                title=f"Top {top_n} Locations by Sales"
                            )
                        else:
                            result_text = "No valid location and sales data available."
                    else:
                        result_text = "Location column or Sales column not found."

                # WORST PRODUCTS
                elif intent == "worst_products":
                    if product_col and sales_col:
                        top_n = extract_top_n(user_query)
                        temp = chat_df[[product_col, sales_col]].copy()
                        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                        temp = temp.dropna()

                        if not temp.empty:
                            grouped = (
                                temp.groupby(product_col, as_index=False)[sales_col]
                                .sum()
                                .sort_values(sales_col, ascending=True)
                                .head(top_n)
                            )

                            result_df = grouped
                            result_text = f"Worst {top_n} Products"
                            names = grouped[product_col].astype(str).tolist()
                            if len(names) == 1:
                                result_summary = f"The lowest-performing product is {names[0]}."
                            else:
                                result_summary = f"The lowest-performing products are " + ", ".join(
                                    names[:-1]) + f", and {names[-1]}."

                            result_fig = px.bar(
                                grouped,
                                x=product_col,
                                y=sales_col,
                                title=f"Worst {top_n} Products"
                            )
                        else:
                            result_text = "No matching data found for that product query."
                    else:
                        result_text = "Product column or Sales column not found."

                # WORST LOCATIONS
                elif intent == "worst_locations":
                    if location_col and sales_col:
                        top_n = extract_top_n(user_query)
                        temp = chat_df[[location_col, sales_col]].copy()
                        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                        temp = temp.dropna()

                        if not temp.empty:
                            grouped = (
                                temp.groupby(location_col, as_index=False)[sales_col]
                                .sum()
                                .sort_values(sales_col, ascending=True)
                                .head(top_n)
                            )

                            result_df = grouped
                            result_text = f"Worst {top_n} Locations"
                            names = grouped[location_col].astype(str).tolist()
                            if len(names) == 1:
                                result_summary = f"The lowest-performing location is {names[0]}."
                            else:
                                result_summary = f"The lowest-performing locations are " + ", ".join(
                                    names[:-1]) + f", and {names[-1]}."

                            result_fig = px.bar(
                                grouped,
                                x=location_col,
                                y=sales_col,
                                title=f"Worst {top_n} Locations by Sales"
                            )
                        else:
                            result_text = "No valid location and sales data available."
                    else:
                        result_text = "Location column or Sales column not found."

                # PROFIT TREND
                elif intent == "profit_trend":
                    if date_col and profit_col:
                        if date_col == profit_col:
                            result_text = "Date column and Profit column cannot be the same."
                        else:
                            temp = chat_df.loc[:, [date_col, profit_col]].copy()
                            temp = temp.loc[:, ~temp.columns.duplicated()]
                            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                            temp[profit_col] = pd.to_numeric(temp[profit_col], errors="coerce")
                            temp = temp.dropna()

                            if not temp.empty:
                                grouped = (
                                    temp.assign(period=temp[date_col].dt.to_period("M").astype(str))
                                    .groupby("period", as_index=False)[profit_col]
                                    .sum()
                                    .sort_values("period")
                                )

                                result_df = grouped
                                result_text = "Profit Trend"
                                result_summary = "This chart shows how profit changed over time by month."
                                result_fig = px.line(
                                    grouped,
                                    x="period",
                                    y=profit_col,
                                    markers=True,
                                    title="Profit Trend"
                                )
                            else:
                                result_text = "No valid date and profit data available for trend."
                    else:
                        result_text = "Date column or Profit column not found."

                # SALES BY
                elif intent == "sales_by":
                    if location_col and sales_col:
                        temp = chat_df[[location_col, sales_col]].copy()
                        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                        temp = temp.dropna()

                        grouped = (
                            temp.groupby(location_col, as_index=False)[sales_col]
                            .sum()
                            .sort_values(sales_col, ascending=False)
                        )

                        result_df = grouped
                        result_text = "Sales by Location"
                        result_summary = "This shows how sales are distributed across locations."
                        result_fig = px.bar(grouped, x=location_col, y=sales_col)
                    else:
                        result_text = "Location or Sales column not found"

                # PROFIT BY
                elif intent == "profit_by":
                    if product_col and profit_col:
                        temp = chat_df[[product_col, profit_col]].copy()
                        temp[profit_col] = pd.to_numeric(temp[profit_col], errors="coerce")
                        temp = temp.dropna()

                        grouped = (
                            temp.groupby(product_col, as_index=False)[profit_col]
                            .sum()
                            .sort_values(profit_col, ascending=False)
                        )

                        result_df = grouped
                        result_text = "Profit by Product"
                        result_summary = "This shows how profit is distributed across products."
                        result_fig = px.bar(grouped, x=product_col, y=profit_col)
                    else:
                        result_text = "Product or Profit column not found"

                # COMPARE PRODUCTS
                elif intent == "compare_products":
                    if product_col and sales_col:
                        temp = chat_df.copy()
                        products = temp[product_col].dropna().astype(str).unique()

                        selected_products = []
                        for p in products:
                            if p.lower() in query:
                                selected_products.append(p)

                        if len(selected_products) >= 2:
                            temp = temp[temp[product_col].isin(selected_products)]
                            temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
                            temp = temp.dropna()

                            grouped = temp.groupby(product_col, as_index=False)[sales_col].sum()

                            result_df = grouped
                            result_text = f"Comparison: {selected_products[0]} vs {selected_products[1]}"
                            result_summary = "This chart compares selected products based on sales."
                            result_fig = px.bar(grouped, x=product_col, y=sales_col)
                        else:
                            result_text = "Please mention at least two products to compare"
                    else:
                        result_text = "Product or Sales column not found"

                else:
                    result_text = "I couldn’t understand that clearly."

                    result_summary = (
                        "Try asking things like:\n"
                        "- total sales\n"
                        "- top 5 products\n"
                        "- sales trend\n"
                        "- top locations\n"
                        "- compare products"
                    )

                if result_text or result_summary or result_df is not None or result_fig is not None:
                    history_entry = {
                        "question": user_query,
                        "text": result_text,
                        "summary": result_summary,
                        "data": result_df.copy() if result_df is not None else None,
                        "figure": result_fig,
                    }

                    if (
                            not st.session_state["chat_history"]
                            or st.session_state["chat_history"][-1]["question"] != user_query
                    ):
                        st.session_state["chat_history"].append(history_entry)

                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    if result_text:
                        st.success(result_text)

                    if result_summary:
                        st.info(result_summary)

                    if (result_df is not None and not result_df.empty) or result_fig is not None:
                        left_now, right_now = st.columns([1, 2])

                        with left_now:
                            if result_df is not None:
                                st.dataframe(result_df, use_container_width=True)

                        with right_now:
                            if result_fig is not None:
                                st.plotly_chart(
                                    result_fig,
                                    use_container_width=True,
                                    key="current_chat_result_fig"
                                )

        st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: CUSTOM VIEWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="custom_views_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Create Custom Views")

filter_blocks_report = []

if not st.session_state.get("manual_clean_applied"):
    st.warning("Please clean your data first to unlock custom views.")
else:
    add_col, caption_col = st.columns([1, 5])

    with add_col:
        if st.button("＋ Add View"):
            add_filter_section()
            st.rerun()

    with caption_col:
        st.caption("Create multiple filtered views from the current cleaned dataset.")

    for sid in st.session_state["filter_section_ids"]:
        st.markdown("---")
        view_name = st.session_state.get(f"section_name_{sid}", f"Custom View {sid}")

        name_col, header_col = st.columns([2, 4])
        with name_col:
            st.text_input("View Name", key=f"section_name_{sid}", label_visibility="collapsed")
        with header_col:
            st.markdown(f"#### {view_name}")

        filter_header, add_filter_btn = st.columns([6, 1])
        with filter_header:
            st.caption("Filter options")
        with add_filter_btn:
            if st.button("＋", key=f"add_filter_btn_{sid}"):
                add_filter_row(sid)
                st.rerun()

        for fid in st.session_state.get(f"filters_{sid}", [1]):
            col_key = f"section_{sid}_filter_col_{fid}"
            val_key = f"section_{sid}_filter_val_{fid}"

            fc, fv, fx = st.columns([3, 3, 1])

            with fc:
                sel_col = st.selectbox(
                    f"Filter {fid} Column",
                    ["None"] + list(final_df.columns),
                    key=col_key,
                )

            with fv:
                if sel_col != "None":
                    st.selectbox(
                        f"Filter {fid} Value",
                        get_column_filter_options(final_df[sel_col]),
                        key=val_key,
                    )
                else:
                    st.selectbox(
                        f"Filter {fid} Value",
                        ["Select a column first"],
                        disabled=True,
                        key=val_key,
                    )

            with fx:
                st.write("")
                st.write("")
                if len(st.session_state.get(f"filters_{sid}", [1])) > 1:
                    if st.button("❌", key=f"remove_filter_{sid}_{fid}"):
                        remove_filter_row(sid, fid)
                        st.rerun()

        section_df, active_filters = get_section_filtered_df(final_df, sid)

        if active_filters:
            st.write("**Applied Filters**")
            for col, val in active_filters:
                st.info(f"{col} = {val}")
        else:
            st.info("No filters selected in this view.")

        show_kpi = st.checkbox("Show KPI Dashboard", key=f"show_kpi_{sid}")
        kpis = None
        profit_for_report = None
        profit_col_for_report = None

        if show_kpi:
            kpis = section_kpis(section_df, final_df)
            view_profit_col = profit_col_override

            if view_profit_col:
                if st.checkbox(f"Show Total Profit ({view_profit_col})", key=f"show_profit_{sid}"):
                    pv = pd.to_numeric(section_df[view_profit_col], errors="coerce").dropna()
                    if not pv.empty:
                        profit_for_report = round(pv.sum(), 2)
                        profit_col_for_report = view_profit_col

            n_kpi_cols = 5 if profit_col_for_report else 4
            kpi_widgets = st.columns(n_kpi_cols)

            with kpi_widgets[0]:
                st.metric("Filtered Rows", kpis["filtered_rows"])
            with kpi_widgets[1]:
                st.metric("% of Total", f"{kpis['percent_of_total']}%")
            with kpi_widgets[2]:
                st.metric("Numeric Columns", kpis["numeric_col_count"])
            with kpi_widgets[3]:
                if kpis["sum_value"] is not None:
                    st.metric(f"Sum of {kpis['sum_col']}", round(kpis["sum_value"], 2))
                else:
                    st.metric("Sum", "N/A")
            if profit_col_for_report:
                with kpi_widgets[4]:
                    st.metric(f"Total Profit ({profit_col_for_report})", profit_for_report)

        st.write("**Filtered Result**")
        row_count_col, table_col = st.columns([1, 4])

        with row_count_col:
            st.metric("Rows Count", len(section_df))
        with table_col:
            if section_df.empty:
                st.warning("No data found for selected filters")
            else:
                st.dataframe(section_df, use_container_width=True)


        st.download_button(
            "Download Filtered Result",
            data=section_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{view_name.replace(' ', '_').lower()}_filtered.csv",
            mime="text/csv",
            key=f"download_filtered_{sid}",
        )

        # ── Live Chart ──
        st.write("**Live Chart Based on Selected Filters**")
        chart_mode = st.selectbox(
            "Chart Mode",
            ["Bar Chart", "Pie Chart", "Both"],
            key=f"live_chart_mode_{sid}"
        )

        add_below_col, _ = st.columns([2, 4])
        with add_below_col:
            if st.button("＋ Add New View", key=f"add_section_below_{sid}"):
                add_filter_section()
                st.rerun()

        st.metric("Filtered Rows (Exact Segment)", len(section_df))

        bar_fig = None
        pie_fig = None
        chart_data = None
        chart_warning = None

        if section_df.empty:
            chart_warning = "No data available after applying filters."
        else:
            filter_count = len(active_filters)

            if filter_count <= 1:
                used_filter_cols = [col for col, _ in active_filters]
                sales_col_section = detect_sales_column(section_df)

                # Prefer a meaningful column for charting
                candidate_cols = [
                    c for c in section_df.columns
                    if c not in used_filter_cols
                       and c != sales_col_section
                       and section_df[c].dropna().nunique() > 1
                ]

                relation_col = None

                # Prefer product/location/category-like columns
                preferred_keywords = ["product", "item", "category", "location", "city", "region", "branch"]
                for c in candidate_cols:
                    c_lower = str(c).lower()
                    if any(k in c_lower for k in preferred_keywords):
                        relation_col = c
                        break

                # Fallback to first usable column
                if relation_col is None and candidate_cols:
                    relation_col = candidate_cols[0]

                if relation_col and sales_col_section:
                    temp_chart = section_df[[relation_col, sales_col_section]].copy()
                    temp_chart[sales_col_section] = pd.to_numeric(temp_chart[sales_col_section], errors="coerce")
                    temp_chart = temp_chart.dropna()

                    if not temp_chart.empty:
                        chart_data = (
                            temp_chart.groupby(relation_col, as_index=False)[sales_col_section]
                            .sum()
                            .sort_values(sales_col_section, ascending=False)
                            .head(10)
                        )

                        bar_fig = px.bar(
                            chart_data,
                            x=relation_col,
                            y=sales_col_section,
                            title=f"{sales_col_section} by {relation_col}",
                        )

                        if len(chart_data) > 1:
                            pie_fig = px.pie(
                                chart_data,
                                names=relation_col,
                                values=sales_col_section,
                                title=f"{sales_col_section} distribution by {relation_col}",
                            )
                    else:
                        chart_warning = "No usable data available for chart after filtering."

                elif relation_col:
                    chart_data = (
                        section_df[relation_col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .reset_index()
                        .head(10)
                    )
                    chart_data.columns = [relation_col, "rows_count"]

                    bar_fig = px.bar(
                        chart_data,
                        x=relation_col,
                        y="rows_count",
                        title=f"Rows Count by {relation_col}",
                    )

                    if len(chart_data) > 1:
                        pie_fig = px.pie(
                            chart_data,
                            names=relation_col,
                            values="rows_count",
                            title=f"Rows Distribution by {relation_col}",
                        )

                else:
                    label = " | ".join(f"{c}={v}" for c, v in active_filters) or "All Data"
                    chart_data = pd.DataFrame({
                        "selection": [label],
                        "rows_count": [len(section_df)],
                    })

                    bar_fig = px.bar(
                        chart_data,
                        x="selection",
                        y="rows_count",
                        title="Rows Count of Current Selection",
                    )

                    if len(section_df) > 0:
                        pie_fig = px.pie(
                            chart_data,
                            names="selection",
                            values="rows_count",
                            title="Rows Distribution",
                        )

            else:
                label = " | ".join(f"{c}={v}" for c, v in active_filters)
                selected_rows = len(section_df)
                remaining_rows = max(len(final_df) - selected_rows, 0)

                chart_data = pd.DataFrame({
                    "segment": [label, "Remaining Data"],
                    "rows_count": [selected_rows, remaining_rows],
                })

                bar_fig = px.bar(
                    chart_data,
                    x="segment",
                    y="rows_count",
                    title="Selected Filters vs Remaining Data",
                )

                if (selected_rows + remaining_rows) > 0:
                    pie_fig = px.pie(
                        chart_data,
                        names="segment",
                        values="rows_count",
                        title="Selected Filters vs Remaining Data",
                    )

        st.session_state["filter_charts"][sid] = {
            "figure": bar_fig if bar_fig is not None else pie_fig,
            "chart_result": chart_data,
            "warning": chart_warning,
        }

        st.write("**View Chart**")
        if active_filters:
            st.caption("Active Filters: " + " | ".join(f"{c} = {v}" for c, v in active_filters))

        if chart_warning:
            st.warning(chart_warning)
        elif chart_data is not None:
            data_col, chart_col = st.columns([1, 2])

            with data_col:
                st.dataframe(chart_data, use_container_width=True)

            with chart_col:
                if chart_mode in ("Bar Chart", "Both") and bar_fig is not None:
                    st.plotly_chart(bar_fig, use_container_width=True)
                if chart_mode in ("Pie Chart", "Both") and pie_fig is not None:
                    st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("Adjust filters to see chart.")

        report_kpis = {}
        if kpis is not None:
            report_kpis = {
                "Filtered Rows": kpis["filtered_rows"],
                "% of Total": f"{kpis['percent_of_total']}%",
                "Numeric Columns": kpis["numeric_col_count"],
                f"Sum of {kpis['sum_col']}" if kpis["sum_col"] else "Sum":
                    round(kpis["sum_value"], 2) if kpis["sum_value"] is not None else "N/A",
            }
            if profit_col_for_report is not None:
                report_kpis[f"Total Profit ({profit_col_for_report})"] = profit_for_report

        filter_blocks_report.append({
            "name": view_name,
            "applied_filters": active_filters,
            "dataframe": section_df,
            "kpis": report_kpis,
        })

st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="insights_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Analyze Your Data")

if not st.session_state.get("manual_clean_applied"):
    st.warning("Please clean your data first to unlock analysis.")
else:
    ic1, ic2, ic3 = st.columns([3, 2, 2])

    with ic1:
        analysis_col = st.selectbox("Select analysis column", final_df.columns, key="analysis_column_main")
    with ic2:
        chart_type = st.selectbox("Choose chart type", ["Bar Chart", "Pie Chart"], key="chart_type_main")
    with ic3:
        top_n = st.number_input("Top results to show", min_value=1, max_value=100, value=10, step=1, key="top_n_main")

    if st.button("Generate Insight Chart"):
        series = final_df[analysis_col]
        numeric_s = pd.to_numeric(series, errors="coerce").dropna()

        display_s = (
            series.dropna().dt.strftime("%Y-%m-%d")
            if pd.api.types.is_datetime64_any_dtype(series)
            else series.dropna().astype(str)
        )

        result = display_s.value_counts().reset_index().head(int(top_n))
        result.columns = [analysis_col, "count"]

        fig = None
        warning = None

        if result.empty:
            warning = "No values available for chart."
        elif chart_type == "Pie Chart":
            if len(result) <= 1:
                warning = "Pie chart is not meaningful for one category."
            else:
                fig = px.pie(
                    result.head(8),
                    names=analysis_col,
                    values="count",
                    title=f"Top {top_n} values in {analysis_col}",
                )
        else:
            fig = px.bar(
                result,
                x=analysis_col,
                y="count",
                title=f"Top {top_n} values in {analysis_col}",
            )

        st.session_state["insight_output"] = {
            "analysis_column": analysis_col,
            "chart_type": chart_type,
            "top_n": int(top_n),
            "figure": fig,
            "result": result,
            "warning": warning,
            "highest_value": numeric_s.max() if not numeric_s.empty else None,
            "lowest_value": numeric_s.min() if not numeric_s.empty else None,
            "highest_count": int((numeric_s == numeric_s.max()).sum()) if not numeric_s.empty else None,
            "lowest_count": int((numeric_s == numeric_s.min()).sum()) if not numeric_s.empty else None,
        }

    if st.session_state["insight_output"] is not None:
        out = st.session_state["insight_output"]

        if out["warning"]:
            st.warning(out["warning"])

        if out["highest_value"] is not None:
            va, vb = st.columns(2)
            with va:
                st.metric(f"Highest value in {out['analysis_column']}", out["highest_value"])
                st.caption(f"Count: {out['highest_count']}")
            with vb:
                st.metric(f"Lowest value in {out['analysis_column']}", out["lowest_value"])
                st.caption(f"Count: {out['lowest_count']}")

        result_col, chart_col = st.columns([1, 2])
        with result_col:
            st.write("**Analysis Result**")
            if out["result"] is not None:
                st.dataframe(out["result"], use_container_width=True)
        with chart_col:
            if out["figure"] is not None:
                st.plotly_chart(out["figure"], use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: COMPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="compare_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Compare Performance")

if not st.session_state.get("manual_clean_applied"):
    st.warning("Please clean your data first to unlock comparison.")
else:
    st.checkbox("Use first custom view for comparison", key="use_filtered_data_for_compare")

    cmp_df = compare_source_df(final_df)
    ensure_compare_col_keys(cmp_df)

    if len(cmp_df.columns) > 0:
        cc1, cc2, cc3 = st.columns([3, 3, 2])

        with cc1:
            compare_col1 = st.selectbox("Select first compare column", cmp_df.columns, key="compare_col1")
        with cc2:
            compare_col2 = st.selectbox("Select second compare column", cmp_df.columns, key="compare_col2")
        with cc3:
            compare_chart_type = st.selectbox(
                "Choose compare chart type",
                ["Bar Chart", "Grouped Bar Chart"],
                key="compare_chart_type",
            )

        if st.button("Create Compare Chart"):
            warning = None
            result = None
            fig = None

            if compare_col1 == compare_col2:
                warning = "Please select two different columns for comparison."
            else:
                def _to_str_series(s: pd.Series) -> pd.Series:
                    return s.dt.strftime("%Y-%m-%d") if pd.api.types.is_datetime64_any_dtype(s) else s.astype(str)

                temp = pd.DataFrame({
                    compare_col1: _to_str_series(cmp_df[compare_col1]),
                    compare_col2: _to_str_series(cmp_df[compare_col2]),
                }).replace("nan", pd.NA).dropna()

                result = temp.groupby([compare_col1, compare_col2]).size().reset_index(name="count")

                if result.empty:
                    warning = "No data available for comparison."
                else:
                    barmode = "group" if compare_chart_type == "Grouped Bar Chart" else None
                    title = f"{'Grouped c' if barmode else 'C'}omparison of {compare_col1} and {compare_col2}"
                    fig = px.bar(
                        result,
                        x=compare_col1,
                        y="count",
                        color=compare_col2,
                        barmode=barmode,
                        title=title,
                    )

            st.session_state["compare_output"] = {
                "compare_col1": compare_col1,
                "compare_col2": compare_col2,
                "compare_chart_type": compare_chart_type,
                "figure": fig,
                "result": result,
                "warning": warning,
            }

    if st.session_state["compare_output"] is not None:
        out = st.session_state["compare_output"]

        if out["warning"]:
            st.warning(out["warning"])

        if out["result"] is not None:
            data_col, chart_col = st.columns([1, 2])

            with data_col:
                st.write("**Compare Data**")
                st.dataframe(out["result"], use_container_width=True)

            with chart_col:
                if out["figure"] is not None:
                    st.plotly_chart(out["figure"], use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION: DOWNLOAD PDF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div id="download_section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Download Full Report")

if not st.session_state.get("manual_clean_applied"):
    st.warning("Please clean your data first to unlock report export.")
else:
    section_figures = [
        {
            "title": st.session_state.get(f"section_name_{sid}", f"Custom View {sid}"),
            "figure": get_section_figure(sid),
        }
        for sid in st.session_state["filter_section_ids"]
        if get_section_figure(sid) is not None
    ]

    report_payload = {
        "original_count": original_row_count,
        "cleaned_count": len(final_df),
        "removed_count": original_row_count - len(final_df),
        "dropped_column_count": len(dropped_columns),
        "columns_to_drop": dropped_columns,
        "remaining_columns": list(final_df.columns),
        "filter_blocks": filter_blocks_report,
        "filter_figures": section_figures,
        "insight_output": st.session_state.get("insight_output"),
        "compare_output": st.session_state.get("compare_output"),
        "insight_figure": get_insight_figure(),
        "compare_figure": get_compare_figure(),
        "smart_insights": generate_smart_insights(final_df) if not final_df.empty else [],
        "chat_history": st.session_state.get("chat_history", []),
    }

    pdf_buffer = create_full_report_pdf(report_payload)

    _, download_col = st.columns([5, 1])
    with download_col:
        st.download_button(
            label="⬇ Download PDF",
            data=pdf_buffer,
            file_name="full_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.caption("Download the full analysis in a single PDF.")

st.markdown('</div>', unsafe_allow_html=True)