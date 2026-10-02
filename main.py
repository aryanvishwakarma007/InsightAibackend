from io import BytesIO
import json
import math
import os
import re
import uuid
from datetime import datetime
from urllib import request as urlrequest


import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="InsightAI API",
    description="AI-powered Excel and CSV Analyzer",
    version="6.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# DETECTION RULES
# ---------------------------------------------------------

ID_PATTERNS = [
    r"(^id$)",
    r"(^id[_\s-])",
    r"([_\s-]id$)",
    r"(^order[_\s-]?id$)",
    r"(^customer[_\s-]?id$)",
    r"(^employee[_\s-]?id$)",
    r"(^product[_\s-]?id$)",
    r"(^invoice[_\s-]?id$)",
    r"(^transaction[_\s-]?id$)",
    r"(^user[_\s-]?id$)",
    r"(^account[_\s-]?id$)",
    r"(^serial[_\s-]?number$)",
    r"(^sku$)",
    r"(^sku[_\s-])",
    r"([_\s-]sku$)",
    r"(^zip[_\s-]?code$)",
    r"(^postal[_\s-]?code$)",
    r"(^pincode$)",
    r"(^pin[_\s-]?code$)",
]

GENERATED_INDEX_NAMES = {
    "index",
    "unnamed: 0",
    "unnamed:0",
    "row index",
    "row_index",
    "row number",
    "row_number",
}

METRIC_PRIORITY = [
    "revenue",
    "sales",
    "amount",
    "profit",
    "income",
    "salary",
    "total",
    "price",
    "quantity",
    "units",
    "score",
    "marks",
    "stock",
    "balance",
    "cost",
]

SECONDARY_METRIC_PRIORITY = [
    "profit",
    "revenue",
    "sales",
    "amount",
    "quantity",
    "units",
    "discount",
    "cost",
    "stock",
    "score",
    "marks",
]

CATEGORY_PRIORITY = [
    "category",
    "department",
    "product",
    "brand",
    "course",
    "subject",
    "type",
    "segment",
    "region",
    "city",
    "state",
    "country",
    "warehouse",
    "location",
    "name",
]

REGION_PRIORITY = [
    "state",
    "province",
    "region",
    "city",
    "country",
    "zone",
    "area",
    "location",
    "warehouse",
]

DATE_PRIORITY = [
    "date",
    "datetime",
    "timestamp",
    "time",
    "month",
    "year",
    "day",
]

STATE_KEYWORDS = [
    "state",
    "province",
    "ship-state",
    "ship state",
    "shipping state",
    "billing state",
    "customer state",
    "delivery state",
]

LOCATION_KEYWORDS = [
    "state",
    "province",
    "city",
    "country",
    "region",
    "location",
    "warehouse",
    "ship-state",
    "ship state",
    "shipping state",
    "billing state",
    "customer state",
    "delivery state",
]


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------


def normalized_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (float, int)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        if numeric.is_integer():
            return int(numeric)
        return round(numeric, 4)

    return str(value)


def clean_records(df):
    records = []
    for _, row in df.iterrows():
        records.append(
            {str(column): clean_value(row[column]) for column in df.columns}
        )
    return records


def format_number(value):
    if value is None:
        return "0"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def is_generated_index_column(column, series):
    """Detect spreadsheet-generated index/placeholder columns consistently."""
    name = normalized_name(column)
    non_null = series.dropna()
    missing_ratio = float(series.isna().mean()) if len(series) else 1.0

    # Any column explicitly named Unnamed:* is treated as a generated spreadsheet
    # artifact when it is sparse. This prevents it leaking into quality metrics.
    if name.startswith("unnamed "):
        if len(non_null) == 0 or missing_ratio >= 0.05:
            return True
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() >= 0.99:
            values = numeric.tolist()
            if values == list(range(len(values))) or values == list(range(1, len(values) + 1)):
                return True

    if name in {normalized_name(x) for x in GENERATED_INDEX_NAMES}:
        if len(non_null) == 0:
            return True
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() >= 0.99:
            values = numeric.tolist()
            if values == list(range(len(values))) or values == list(range(1, len(values) + 1)):
                return True

    return False


def looks_like_id(column, series):
    name = normalized_name(column)

    for pattern in ID_PATTERNS:
        if re.search(pattern, str(column).strip().lower()):
            return True

    # Generic ID/code/key/reference names.
    if any(
        token in name.split()
        for token in ["identifier", "reference", "ref", "key", "code", "number"]
    ):
        # Avoid treating meaningful measures such as phone numbers or prices as IDs
        # solely because they contain "number".
        if not any(word in name for word in ["quantity", "price", "amount", "number of"]):
            return True

    # Business measures should never be treated as IDs just because they are
    # high-cardinality or monotonically increasing (for example Amount).
    measure_tokens = {
        "revenue", "sales", "amount", "profit", "income", "salary",
        "total", "price", "quantity", "qty", "units", "score",
        "marks", "stock", "balance", "cost", "discount"
    }
    if any(token in name.split() for token in measure_tokens):
        return False

    non_null = series.dropna()
    if len(non_null) <= 20:
        return False

    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)

    if unique_ratio >= 0.995:
        if pd.api.types.is_float_dtype(series):
            return False

        if pd.api.types.is_integer_dtype(series):
            return bool(non_null.is_monotonic_increasing)

        return True

    return False


def detect_date_column(df, column):
    """Detect real date/time columns using both column semantics and parse quality."""
    series = df[column]

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Numeric columns are kept out here. A column containing 2022/2023 or
    # Excel-like numeric IDs should not silently become a date axis.
    if pd.api.types.is_numeric_dtype(series):
        return False

    name = normalized_name(column)
    tokens = set(name.split())
    name_suggests_date = bool(tokens & set(DATE_PRIORITY)) or any(
        key in name for key in ("created at", "updated at", "order date", "invoice date", "hire date", "birth date", "dob")
    )

    sample = series.dropna().astype(str).str.strip().head(1000)
    if len(sample) < 2:
        return False

    # Reject columns that are almost entirely IDs/codes even when a few values
    # happen to parse as dates.
    id_like = any(token in tokens for token in ("id", "code", "sku", "zip", "postal", "pin"))
    if id_like and not name_suggests_date:
        return False

    parsed_default = pd.to_datetime(sample, errors="coerce", format="mixed")
    parsed_dayfirst = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=True)
    default_ratio = float(parsed_default.notna().mean())
    dayfirst_ratio = float(parsed_dayfirst.notna().mean())
    valid_ratio = max(default_ratio, dayfirst_ratio)

    # Date-looking names can tolerate a small amount of dirty data; generic
    # text columns need very strong evidence before becoming a date column.
    threshold = 0.65 if name_suggests_date else 0.94
    if valid_ratio < threshold:
        return False

    # A date column should contain more than one distinct parsed date.
    parsed = parsed_default if default_ratio >= dayfirst_ratio else parsed_dayfirst
    return parsed.dropna().nunique() >= 2


def is_flag_column(column, series):
    """Return True for boolean/flag-like fields that should not become business metrics."""
    name = normalized_name(column)
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    unique = non_null.nunique(dropna=True)
    if unique <= 2:
        values = {str(v).strip().lower() for v in non_null.head(5000)}
        flag_names = (
            "b2b", "flag", "is_", "has_", "active", "status",
            "returned", "cancelled", "verified", "approved", "enabled"
        )
        if any(token in name for token in flag_names):
            return True
        if values.issubset({"0", "1", "true", "false", "yes", "no", "y", "n"}):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return True
    return False


def _metric_score(column):
    name = normalized_name(column)
    score = 0
    for rank, keyword in enumerate(METRIC_PRIORITY):
        if keyword in name.split() or keyword in name:
            score += 100 - rank * 5
    if any(token in name.split() for token in ["id", "code", "key", "reference", "number"]):
        score -= 160
    return score


def choose_metric(numeric_columns, df=None, identifier_columns=None, date_columns=None):
    if not numeric_columns:
        return None
    excluded = set(identifier_columns or []) | set(date_columns or [])
    candidates = [
        c for c in numeric_columns
        if c not in excluded and (df is None or not is_flag_column(c, df[c]))
    ]
    if not candidates:
        return None

    scored = []
    for column in candidates:
        series = pd.to_numeric(df[column], errors="coerce") if df is not None else pd.Series(dtype=float)
        score = _metric_score(column)
        if df is not None and len(series):
            valid = series.dropna()
            if valid.empty:
                score -= 100
            else:
                unique_ratio = valid.nunique() / max(len(valid), 1)
                if unique_ratio > 0.98:
                    score -= 45
                if valid.nunique() <= 1:
                    score -= 100
                if (valid >= 0).mean() > 0.98:
                    score += 4
        scored.append((score, column))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def choose_secondary_metric(numeric_columns, primary, df=None, identifier_columns=None, date_columns=None):
    """Choose a genuine business measure, not flags, IDs, codes or tiny-cardinality fields."""
    excluded = set(identifier_columns or []) | set(date_columns or [])
    others = [
        column for column in numeric_columns
        if column != primary
        and column not in excluded
        and (df is None or not is_flag_column(column, df[column]))
    ]
    if not others:
        return None

    preferred = [
        "profit", "revenue", "sales", "amount", "quantity", "qty",
        "units", "discount", "cost", "stock", "score", "marks", "salary"
    ]
    blocked_tokens = {"id", "code", "key", "flag", "b2b", "zip", "postal", "pin"}
    ranked = []
    for column in others:
        name = normalized_name(column)
        tokens = set(name.split())
        score = 0
        for rank, keyword in enumerate(preferred):
            if keyword in tokens or keyword in name:
                score += 100 - rank * 5
        if tokens & blocked_tokens:
            score -= 200
        if df is not None:
            valid = pd.to_numeric(df[column], errors="coerce").dropna()
            unique_count = int(valid.nunique())
            if valid.empty or unique_count <= 1:
                score -= 300
            else:
                unique_ratio = unique_count / max(len(valid), 1)
                if unique_ratio > 0.995:
                    score -= 35
                # A numeric field with <= 10 distinct values is normally a flag/code
                # unless its name explicitly describes a useful measure.
                measure_named = any(k in name for k in preferred)
                if unique_count <= 10 and not measure_named:
                    score -= 180
        ranked.append((score, column))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1] if ranked and ranked[0][0] > -100 else None


def choose_category(category_columns, df=None):
    if not category_columns:
        return None
    ranked = []
    for column in category_columns:
        name = normalized_name(column)
        unique = int(df[column].nunique(dropna=True)) if df is not None else 0
        score = 0
        for rank, keyword in enumerate(CATEGORY_PRIORITY):
            if keyword in name.split() or keyword in name:
                score += 100 - rank * 4
        # Prefer useful business dimensions over nearly-unique labels.
        if unique <= 1:
            score -= 100
        elif unique <= 30:
            score += 25
        elif unique <= 100:
            score += 10
        else:
            score -= 25
        if any(token in name.split() for token in ["id", "code", "sku", "reference"]):
            score -= 100
        ranked.append((score, column))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1]

def choose_region(category_columns, primary):
    others = [column for column in category_columns if column != primary]

    for keyword in REGION_PRIORITY:
        for column in others:
            name = normalized_name(column)
            if keyword in name.split() or keyword in name:
                return column

    return None


def choose_location_column(text_columns, category_columns):
    candidates = list(dict.fromkeys(category_columns + text_columns))

    # Prefer explicit state/province columns.
    for keyword in STATE_KEYWORDS:
        for column in candidates:
            name = normalized_name(column)
            if keyword in name:
                return column, "state"

    for keyword in LOCATION_KEYWORDS:
        for column in candidates:
            name = normalized_name(column)
            if keyword in name:
                return column, "location"

    return None, None


# ---------------------------------------------------------
# STATE / LOCATION NORMALIZATION
# ---------------------------------------------------------

STATE_CANONICAL = {
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman nicobar": "Andaman and Nicobar Islands",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jammu and kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "west bengal": "West Bengal",
}


def normalize_location(value, location_type):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return None

    if location_type == "state":
        key = normalized_name(text)
        return STATE_CANONICAL.get(key, text.title())

    return text.title()


def aggregate_location(df, location_column, metric, quantity_column=None, location_type="location"):
    if not location_column:
        return []

    work = pd.DataFrame()
    work[location_column] = df[location_column].map(
        lambda value: normalize_location(value, location_type)
    )
    work = work.dropna(subset=[location_column])

    if work.empty:
        return []

    if metric:
        work[metric] = pd.to_numeric(df.loc[work.index, metric], errors="coerce")
        work = work.dropna(subset=[metric])

        grouped = (
            work.groupby(location_column)[metric]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
    else:
        grouped = work[location_column].value_counts().head(10)

    result = []
    for label, value in grouped.items():
        result.append(
            {
                "label": str(label),
                "value": clean_value(value),
                "display": format_number(value),
            }
        )

    return result


# ---------------------------------------------------------
# DATASET IDENTITY
# ---------------------------------------------------------


def detect_dataset_identity(df, columns):
    names = " ".join(normalized_name(column) for column in columns)

    rules = [
        (
            ["pubg", "free fire", "freefire", "kills", "damage", "match", "weapon", "player"],
            "Gaming",
            "🎮",
            "Gaming Analytics",
            "Game, player or match performance data",
        ),
        (
            ["order", "sku", "quantity", "amount", "sales", "product", "invoice", "price"],
            "E-commerce",
            "🛒",
            "E-commerce Sales",
            "Orders, products, sales and transaction data",
        ),
        (
            ["student", "marks", "grade", "subject", "course", "attendance"],
            "Education",
            "🎓",
            "Education Analytics",
            "Student, course, marks or attendance data",
        ),
        (
            ["employee", "salary", "department", "hire date", "job title", "designation"],
            "HR",
            "👥",
            "HR / Employee Analytics",
            "Employee, salary, department or hiring data",
        ),
        (
            ["stock", "inventory", "warehouse", "reorder", "sku"],
            "Inventory",
            "📦",
            "Inventory Analytics",
            "Stock, product and warehouse data",
        ),
        (
            ["transaction", "balance", "account", "credit", "debit", "loan", "interest"],
            "Finance",
            "💰",
            "Finance Analytics",
            "Financial transaction or account data",
        ),
        (
            ["patient", "diagnosis", "hospital", "doctor", "medical", "disease"],
            "Healthcare",
            "🏥",
            "Healthcare Analytics",
            "Patient, medical or hospital data",
        ),
        (
            ["campaign", "click", "impression", "conversion", "lead", "marketing"],
            "Marketing",
            "📣",
            "Marketing Analytics",
            "Campaign, traffic or conversion data",
        ),
        (
            ["shipment", "delivery", "carrier", "shipping", "warehouse", "tracking"],
            "Logistics",
            "🚚",
            "Logistics Analytics",
            "Shipment, delivery or logistics data",
        ),
    ]

    best = None
    best_score = 0

    for keywords, category, icon, title, description in rules:
        score = 0
        for keyword in keywords:
            if normalized_name(keyword) in names:
                score += 1

        if score > best_score:
            best_score = score
            best = (category, icon, title, description)

    if best:
        category, icon, title, description = best
        # Confidence is based on the winning rule's own keyword count, not the
        # last rule iterated through. Keep the label honest when only one signal exists.
        winning_keywords = next(
            (keywords for keywords, rule_category, *_ in rules if rule_category == category),
            [],
        )
        # Use a bounded evidence scale: one matching signal is not 100% confidence.
        if best_score <= 1:
            confidence = 55
        elif best_score == 2:
            confidence = 70
        elif best_score == 3:
            confidence = 85
        else:
            confidence = 95
        return {
            "category": category,
            "icon": icon,
            "title": title,
            "description": description,
            "confidence": confidence,
        }

    return {
        "category": "General",
        "icon": "📊",
        "title": "Data Analytics",
        "description": "General structured data analysis",
        "confidence": 40,
    }


# ---------------------------------------------------------
# CHART AGGREGATION
# ---------------------------------------------------------


def aggregate_category(df, category, metric):
    if not category:
        return []

    work = df[[category]].copy()

    if metric:
        work[metric] = pd.to_numeric(df[metric], errors="coerce")
        grouped = (
            work.dropna(subset=[category])
            .groupby(category, dropna=False)[metric]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )

        return [
            {
                "label": str(label),
                "value": clean_value(value),
                "display": format_number(value),
            }
            for label, value in grouped.items()
        ]

    grouped = work[category].dropna().astype(str).value_counts().head(8)

    return [
        {
            "label": str(label),
            "value": int(value),
            "display": format_number(value),
        }
        for label, value in grouped.items()
    ]


def parse_dates(series):
    """Parse common real-world date formats while avoiding invented dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    raw = series.copy()
    parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    fallback = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=True)

    # Prefer the parser that successfully interprets more rows. When both
    # parse equally well, preserve pandas' default ordering instead of making
    # an arbitrary day/month decision.
    if fallback.notna().sum() > parsed.notna().sum():
        parsed = fallback

    # A small set of common explicit formats handles spreadsheets whose mixed
    # parser is conservative (for example "31/12/2025 14:30").
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y",
        "%d/%m/%y", "%d-%m-%y", "%m/%d/%y", "%m-%d-%y",
        "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    ]
    best = parsed
    best_count = int(parsed.notna().sum())
    for fmt in formats:
        candidate = pd.to_datetime(raw, errors="coerce", format=fmt)
        count = int(candidate.notna().sum())
        if count > best_count:
            best, best_count = candidate, count

    return best


def aggregate_trend(df, date_column, metric):
    if not date_column or not metric:
        return []

    work = df[[date_column, metric]].copy()
    work[date_column] = parse_dates(work[date_column])
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[date_column, metric])

    if work.empty:
        return []

    daily = (
        work.groupby(work[date_column].dt.to_period("D"))[metric]
        .sum()
        .reset_index()
    )

    grouped = daily
    frequency = "daily"

    # Large date ranges are easier to read as monthly totals.
    if len(grouped) > 30:
        grouped = (
            work.groupby(work[date_column].dt.to_period("M"))[metric]
            .sum()
            .reset_index()
        )
        frequency = "monthly"

    if len(grouped) > 30:
        grouped = grouped.tail(30)

    result = []
    for _, row in grouped.iterrows():
        result.append(
            {
                "label": str(row[date_column]),
                "value": clean_value(row[metric]),
                "display": format_number(row[metric]),
            }
        )

    return result


def calculate_distribution(category_data):
    if not category_data:
        return []

    total = sum(float(item.get("value") or 0) for item in category_data)
    if total <= 0:
        return []

    return [
        {
            **item,
            "percentage": round(
                float(item.get("value") or 0) / total * 100,
                1,
            ),
        }
        for item in category_data
    ]


# ---------------------------------------------------------
# ADVANCED ANALYTICS
# ---------------------------------------------------------


def build_missing_by_column(df, limit=10):
    total_rows = max(len(df), 1)
    rows = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            rows.append({
                "column": str(column),
                "missing": missing,
                "percentage": round(missing / total_rows * 100, 1),
            })
    rows.sort(key=lambda x: (-x["missing"], x["column"]))
    return rows[:limit]


def build_outlier_summary(df, numeric_columns, limit=8):
    results = []
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 8 or values.nunique() < 4:
            continue
        sample = values.sample(min(len(values), 100_000), random_state=42) if len(values) > 100_000 else values
        q1 = float(sample.quantile(0.25))
        q3 = float(sample.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            results.append({
                "column": str(column),
                "count": count,
                "percentage": round(count / len(values) * 100, 1),
                "lower": clean_value(lower),
                "upper": clean_value(upper),
            })
    results.sort(key=lambda x: (-x["count"], x["column"]))
    return results[:limit]


def build_correlations(df, numeric_columns, limit=8):
    usable = [c for c in numeric_columns if df[c].nunique(dropna=True) >= 3]
    if len(usable) < 2:
        return []
    sample = df[usable].copy()
    if len(sample) > 100_000:
        sample = sample.sample(100_000, random_state=42)
    corr = sample.corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append({
                "x": str(left),
                "y": str(right),
                "correlation": round(float(value), 3),
                "strength": "strong" if abs(value) >= 0.7 else "moderate" if abs(value) >= 0.4 else "weak",
            })
    pairs.sort(key=lambda x: -abs(x["correlation"]))
    return pairs[:limit]


def build_anomalies(trend_data, metric, limit=5):
    if not metric or len(trend_data) < 6:
        return []
    values = pd.Series([float(x.get("value") or 0) for x in trend_data], dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 0:
        return []
    rows = []
    for index, row in enumerate(trend_data):
        z = (float(row.get("value") or 0) - mean) / std
        if abs(z) >= 2:
            rows.append({
                "label": row.get("label"),
                "value": row.get("value"),
                "display": row.get("display"),
                "z_score": round(float(z), 2),
                "direction": "high" if z > 0 else "low",
            })
    rows.sort(key=lambda x: -abs(x["z_score"]))
    return rows[:limit]


def build_dynamic_kpis(df, metric, metric2, category, location_column):
    result = []
    if metric and not is_flag_column(metric, df[metric]):
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(values):
            result.append({"label": f"Total {metric}", "value": clean_value(values.sum()), "display": format_number(values.sum()), "type": "metric"})
            result.append({"label": f"Average {metric}", "value": clean_value(values.mean()), "display": format_number(values.mean()), "type": "average"})
            result.append({"label": f"Highest {metric}", "value": clean_value(values.max()), "display": format_number(values.max()), "type": "max"})
    if metric2 and metric2 != metric and not is_flag_column(metric2, df[metric2]):
        values = pd.to_numeric(df[metric2], errors="coerce").dropna()
        if len(values):
            result.append({"label": f"Total {metric2}", "value": clean_value(values.sum()), "display": format_number(values.sum()), "type": "secondary"})
    if category:
        result.append({"label": f"Unique {category}", "value": int(df[category].nunique(dropna=True)), "display": format_number(df[category].nunique(dropna=True)), "type": "dimension"})
    if location_column and location_column != category:
        result.append({"label": f"Unique {location_column}", "value": int(df[location_column].nunique(dropna=True)), "display": format_number(df[location_column].nunique(dropna=True)), "type": "location"})
    return result[:6]


def build_stat_summary(df, numeric_columns, limit=10):
    rows = []
    for column in numeric_columns[:limit]:
        if is_flag_column(column, df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "column": str(column),
            "count": int(values.count()),
            "mean": clean_value(values.mean()),
            "median": clean_value(values.median()),
            "min": clean_value(values.min()),
            "max": clean_value(values.max()),
            "std": clean_value(values.std()),
        })
    return rows


def build_advanced_insights(metric, metric2, outliers, correlations, anomalies, missing_by_column):
    insights = []
    if outliers:
        item = outliers[0]
        insights.append({
            "type": "warning",
            "title": "Potential Outliers",
            "text": f"{item['column']} contains {item['count']:,} potential outlier values ({item['percentage']:.1f}% of non-missing values) using the IQR rule.",
        })
    if correlations:
        item = correlations[0]
        relation = "positive" if item["correlation"] > 0 else "negative"
        insights.append({
            "type": "highlight",
            "title": "Strongest Numeric Relationship",
            "text": f"{item['x']} and {item['y']} show a {relation} correlation of {item['correlation']:.2f}.",
        })
    if anomalies:
        item = anomalies[0]
        insights.append({
            "type": "warning",
            "title": "Trend Anomaly",
            "text": f"{item['label']} is an unusually {item['direction']} period for {metric or 'the primary metric'} (z-score {item['z_score']}).",
        })
    if missing_by_column:
        item = missing_by_column[0]
        insights.append({
            "type": "warning",
            "title": "Most Missing Column",
            "text": f"{item['column']} has {item['missing']:,} missing values ({item['percentage']:.1f}% of rows).",
        })
    return insights


# ---------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------


def build_insights(
    df,
    metric,
    category,
    region,
    location_column,
    location_type,
    date_column,
    category_data,
    trend_data,
    metric_total,
    missing_values,
    missing_percent,
    duplicate_rows,
    rows_with_missing,
):
    insights = []

    # 1. Highest category
    if category_data and metric and metric_total and metric_total > 0:
        top = category_data[0]
        share = float(top["value"] or 0) / metric_total * 100
        insights.append(
            {
                "type": "highlight",
                "title": f"Top {category}",
                "text": (
                    f"{top['label']} has the highest {metric} "
                    f"({top['display']}), contributing {share:.1f}% of the total."
                ),
            }
        )

    # 2. Recent trend. Compare the latest two meaningful periods instead of
    # the first and last period. This prevents tiny early periods from creating
    # misleading values such as +22,000%.
    if date_column and len(trend_data) >= 2:
        previous = float(trend_data[-2]["value"] or 0)
        latest = float(trend_data[-1]["value"] or 0)
        total_abs = abs(float(metric_total or 0))

        # Only calculate a percentage when the comparison period is meaningful.
        # At least 2% of the total is a safer baseline than a tiny first bucket.
        meaningful_baseline = max(total_abs * 0.02, 1.0)

        if abs(previous) >= meaningful_baseline:
            change = (latest - previous) / abs(previous) * 100
            direction = "up" if change >= 0 else "down"

            if abs(change) >= 3:
                insights.append(
                    {
                        "type": "trend",
                        "direction": direction,
                        "title": "Recent Growth" if direction == "up" else "Recent Decline",
                        "text": (
                            f"{metric} {'increased' if direction == 'up' else 'decreased'} "
                            f"by {abs(change):.1f}% from {trend_data[-2]['label']} "
                            f"to {trend_data[-1]['label']}."
                        ),
                    }
                )
            else:
                insights.append(
                    {
                        "type": "trend",
                        "direction": "flat",
                        "title": "Stable Recent Trend",
                        "text": (
                            f"{metric} changed by {abs(change):.1f}% from "
                            f"{trend_data[-2]['label']} to {trend_data[-1]['label']}."
                        ),
                    }
                )
        else:
            # Do not show a giant percentage when the previous period is only a
            # tiny fraction of the overall metric. Report the absolute movement
            # instead and explicitly explain why the percentage is withheld.
            absolute_change = latest - previous
            direction = "up" if absolute_change >= 0 else "down"
            insights.append(
                {
                    "type": "trend",
                    "direction": direction,
                    "title": "Recent Change",
                    "text": (
                        f"{metric} {'increased' if direction == 'up' else 'decreased'} "
                        f"by {format_number(abs(absolute_change))} from "
                        f"{trend_data[-2]['label']} to {trend_data[-1]['label']}. "
                        "A percentage change was withheld because the earlier period "
                        "is too small for a reliable percentage comparison."
                    ),
                }
            )

    # 3. Missing values
    if missing_values > 0:
        insights.append(
            {
                "type": "warning",
                "title": "Attention Needed",
                "text": (
                    f"{missing_values:,} missing cells ({missing_percent:.1f}% of all cells) "
                    f"are spread across {rows_with_missing:,} rows."
                ),
            }
        )

    # 4. Duplicates
    if duplicate_rows > 0:
        insights.append(
            {
                "type": "warning",
                "title": "Duplicate Rows Found",
                "text": (
                    f"{duplicate_rows:,} exact duplicate rows were detected. "
                    "Review them before calculating totals."
                ),
            }
        )

    # 5. Top location using a real metric, never a fake "margin" metric.
    if location_column:
        location_data = aggregate_location(
            df,
            location_column,
            metric,
            location_type=location_type,
        )
        if location_data:
            top_location = location_data[0]
            metric_label = metric or "records"
            insights.append(
                {
                    "type": "region",
                    "title": f"Top {location_type.title()} by {metric_label}",
                    "text": (
                        f"{top_location['label']} has the highest {metric_label} "
                        f"({top_location['display']})."
                    ),
                }
            )

    if missing_values == 0 and duplicate_rows == 0:
        insights.append(
            {
                "type": "quality",
                "title": "Clean Dataset",
                "text": "No missing cells or exact duplicate rows were detected.",
            }
        )

    if not insights:
        insights.append(
            {
                "type": "quality",
                "title": "Basic Analysis Done",
                "text": "The file was analyzed successfully. Add suitable numeric and category fields for richer charts.",
            }
        )

    return insights[:6]


# ---------------------------------------------------------
# SESSION / PRODUCT FEATURES
# ---------------------------------------------------------

# In-memory sessions keep the current upload available for interactive
# filtering, chat and forecasting during local development. Production should
# move this state to Redis/object storage/database.
ANALYSIS_SESSIONS = {}
MAX_SESSIONS = 20


def _session_cleanup():
    while len(ANALYSIS_SESSIONS) > MAX_SESSIONS:
        oldest = next(iter(ANALYSIS_SESSIONS))
        ANALYSIS_SESSIONS.pop(oldest, None)


def _get_session(analysis_id):
    session = ANALYSIS_SESSIONS.get(str(analysis_id))
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session expired. Please upload the file again.")
    return session


def _apply_filters(df, filters):
    work = df.copy()
    filters = filters or {}

    category_column = filters.get("category_column")
    category_value = filters.get("category_value")
    location_column = filters.get("location_column")
    location_value = filters.get("location_value")
    date_column = filters.get("date_column")
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    if category_column and category_column in work.columns and category_value not in (None, "", "All"):
        work = work[work[category_column].astype(str).str.strip() == str(category_value).strip()]

    if location_column and location_column in work.columns and location_value not in (None, "", "All"):
        if "state" in normalized_name(location_column):
            values = work[location_column].map(lambda x: normalize_location(x, "state"))
            work = work[values == normalize_location(location_value, "state")]
        else:
            work = work[work[location_column].astype(str).str.strip().str.title() == str(location_value).strip().title()]

    if date_column and date_column in work.columns and (date_from or date_to):
        parsed = parse_dates(work[date_column])
        if date_from:
            start = pd.to_datetime(date_from, errors="coerce")
            if pd.notna(start):
                work = work[parsed >= start]
        if date_to:
            end = pd.to_datetime(date_to, errors="coerce")
            if pd.notna(end):
                work = work[parsed <= end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]

    return work


def _build_analysis_payload(df, session):
    metric = session["metric"]
    metric2 = session["metric2"]
    category = session["category"]
    date_column = session["date_column"]
    location_column = session["location_column"]
    location_type = session["location_type"]
    identifier_columns = session["identifier_columns"]
    flag_columns = session["flag_columns"]
    generated_columns = session["generated_columns"]

    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    missing_values = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())
    duplicate_rows = int(df.duplicated().sum())

    category_data = aggregate_category(df, category, metric)
    trend_data = aggregate_trend(df, date_column, metric)
    secondary_trend = aggregate_trend(df, date_column, metric2)
    distribution = calculate_distribution(category_data)
    location_data = aggregate_location(df, location_column, metric, location_type=location_type or "location")

    metric_total = None
    if metric and metric in df.columns:
        metric_total = float(pd.to_numeric(df[metric], errors="coerce").sum())
    secondary_total = None
    if metric2 and metric2 in df.columns:
        secondary_total = float(pd.to_numeric(df[metric2], errors="coerce").sum())

    usable_numeric = [c for c in session["numeric_columns"] if c not in flag_columns and c in df.columns]
    missing_by_column = build_missing_by_column(df)
    outlier_summary = build_outlier_summary(df, usable_numeric)
    correlation_summary = build_correlations(df, usable_numeric)
    anomaly_summary = build_anomalies(trend_data, metric)
    dynamic_kpis = build_dynamic_kpis(df, metric, metric2, category, location_column)
    stat_summary = build_stat_summary(df, usable_numeric)
    advanced_insights = build_advanced_insights(metric, metric2, outlier_summary, correlation_summary, anomaly_summary, missing_by_column)

    total_cells = total_rows * total_columns
    completeness = ((total_cells - missing_values) / total_cells * 100) if total_cells else 100
    missing_percent = (missing_values / total_cells * 100) if total_cells else 0
    duplicate_percent = (duplicate_rows / total_rows * 100) if total_rows else 0

    insights = (build_insights(
        df=df, metric=metric, category=category, region=session.get("region"),
        location_column=location_column, location_type=location_type or "location",
        date_column=date_column, category_data=category_data, trend_data=trend_data,
        metric_total=metric_total, missing_values=missing_values, missing_percent=missing_percent,
        duplicate_rows=duplicate_rows, rows_with_missing=rows_with_missing,
    ) + advanced_insights)[:10]

    return {
        "total_rows": total_rows, "total_columns": total_columns,
        "columns": [str(c) for c in df.columns],
        "missing_values": missing_values, "rows_with_missing": rows_with_missing,
        "duplicate_rows": duplicate_rows,
        "metric_total": clean_value(metric_total),
        "metric_total_display": format_number(metric_total) if metric_total is not None else None,
        "secondary_total": clean_value(secondary_total),
        "secondary_total_display": format_number(secondary_total) if secondary_total is not None else None,
        "charts": {"trend": trend_data, "secondary_trend": secondary_trend, "category": category_data, "distribution": distribution, "location": location_data},
        "quality": {
            "completeness": round(completeness,1), "missing_cells": missing_values,
            "missing_percent": round(missing_percent,1), "rows_with_missing": rows_with_missing,
            "duplicate_rows": duplicate_rows, "duplicate_percent": round(duplicate_percent,1),
            "columns_with_missing": len(missing_by_column), "outlier_columns": len(outlier_summary),
        },
        "analytics": {"kpis": dynamic_kpis, "missing_by_column": missing_by_column, "outliers": outlier_summary, "correlations": correlation_summary, "anomalies": anomaly_summary, "stat_summary": stat_summary},
        "insights": insights,
        "filtered": True,
    }


def _resolve_forecast_metric(session):
    df = session.get("df")
    candidates = [session.get("metric"), session.get("metric2")]
    candidates += session.get("numeric_columns", [])
    blocked = set(session.get("identifier_columns", [])) | set(session.get("flag_columns", []))
    for candidate in candidates:
        if not candidate or candidate not in df.columns or candidate in blocked:
            continue
        values = pd.to_numeric(df[candidate], errors="coerce")
        if values.notna().sum() >= 8 and values.nunique(dropna=True) >= 2:
            return candidate
    return None


def _resolve_forecast_date(session):
    df = session.get("df")
    candidates = [session.get("date_column")] + session.get("date_columns", [])
    # Re-check all columns as a fallback. This makes Forecast independent from
    # the upload-time detector if a spreadsheet date column was borderline.
    candidates += [str(c) for c in df.columns]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or candidate not in df.columns:
            continue
        seen.add(candidate)
        parsed = parse_dates(df[candidate])
        if parsed.notna().sum() >= 8 and parsed.dropna().nunique() >= 4:
            return candidate
    return None


def _forecast(df, metric, date_column, periods=6):
    if df is None or df.empty:
        return {"available": False, "reason": "No dataset is available for forecasting."}
    if not metric or metric not in df.columns:
        return {"available": False, "reason": "No usable numeric metric was detected for forecasting."}
    if not date_column or date_column not in df.columns:
        return {"available": False, "reason": "No usable date column was detected for forecasting."}

    work = pd.DataFrame({
        "date": parse_dates(df[date_column]),
        "value": pd.to_numeric(df[metric], errors="coerce"),
    }).dropna()
    if len(work) < 8:
        return {"available": False, "reason": "At least 8 valid date/value observations are needed for a basic forecast."}

    # Prefer monthly aggregation for transaction/business datasets. If fewer
    # than 6 months exist, use daily periods instead.
    monthly = work.groupby(work["date"].dt.to_period("M"))["value"].sum().reset_index()
    if len(monthly) >= 6:
        grouped = monthly
        frequency = "monthly"
    else:
        grouped = work.groupby(work["date"].dt.to_period("D"))["value"].sum().reset_index()
        frequency = "daily"

    if len(grouped) < 6:
        return {"available": False, "reason": "Not enough distinct time periods for a stable forecast."}

    y = grouped["value"].astype(float).to_numpy()
    x = list(range(len(y)))
    degree = 1 if len(y) < 18 else 2
    try:
        import numpy as np
        coef = np.polyfit(x, y, degree)
        model = np.poly1d(coef)
        future_x = list(range(len(y), len(y) + periods))
        pred = [max(0.0, float(model(v))) for v in future_x]
    except Exception:
        return {"available": False, "reason": "Forecast model could not be fitted to this dataset."}

    last_period = grouped["date"].iloc[-1]
    result = []
    for i, value in enumerate(pred, 1):
        period = last_period + i
        result.append({
            "label": str(period),
            "value": clean_value(value),
            "display": format_number(value),
        })

    # Return historical points too, so the frontend can render one continuous
    # visual forecast: actual history + future estimate.
    history = []
    for _, row in grouped.iterrows():
        history.append({
            "label": str(row["date"]),
            "actual": clean_value(row["value"]),
        })

    chart = []
    for item in history:
        chart.append({"label": item["label"], "actual": item["actual"], "forecast": None})
    if history:
        # Connect the forecast line naturally from the final actual point.
        chart[-1]["forecast"] = history[-1]["actual"]
    for item in result:
        chart.append({"label": item["label"], "actual": None, "forecast": item["value"]})

    return {
        "available": True,
        "method": "trend-based forecast",
        "metric": metric,
        "date_column": date_column,
        "frequency": frequency,
        "history": history,
        "chart": chart,
        "forecast": result,
    }


def _chat_answer(session, question):
    """Answer common natural-language questions directly from the uploaded data.

    The optional LLM path receives only a compact dataset summary. Without an API
    key, InsightAI still works as a deterministic data assistant so the portfolio
    project is usable without paid AI infrastructure.
    """
    df = session["df"]
    q = str(question or "").strip()
    if not q:
        return "Please ask a question about the uploaded dataset."

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        summary = {
            "rows": len(df),
            "columns": [str(c) for c in df.columns],
            "primary_metric": session.get("metric"),
            "secondary_metric": session.get("metric2"),
            "category": session.get("category"),
            "date": session.get("date_column"),
            "location": session.get("location_column"),
            "numeric_summary": build_stat_summary(
                df,
                [c for c in session["numeric_columns"] if c not in session["flag_columns"]],
                8,
            ),
        }
        body = json.dumps({
            "model": "gpt-5-mini",
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are InsightAI's data analyst assistant. Answer only from "
                        "the supplied dataset summary. Be concise, factual, and mention "
                        "when the summary is insufficient. Do not invent values."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"question": q, "dataset": summary}, ensure_ascii=False),
                },
            ],
        }).encode()
        try:
            req = urlrequest.Request(
                "https://api.openai.com/v1/responses",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode())
            text = payload.get("output_text")
            if text:
                return text.strip()
        except Exception:
            # Never break the dashboard because an optional external AI service fails.
            pass

    low = q.lower()
    metric = session.get("metric")
    metric2 = session.get("metric2")
    category = session.get("category")
    location = session.get("location_column")
    date_column = session.get("date_column")

    def numeric_values(column):
        if not column or column not in df.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(df[column], errors="coerce").dropna()

    # Dataset-level questions.
    if any(k in low for k in ["how many rows", "number of rows", "total rows", "records"]):
        return f"The dataset contains {len(df):,} rows and {len(df.columns):,} analyzed columns."
    if any(k in low for k in ["how many columns", "number of columns", "total columns"]):
        return f"The dataset contains {len(df.columns):,} analyzed columns."
    if "duplicate" in low:
        return f"The dataset contains {int(df.duplicated().sum()):,} exact duplicate rows."
    if "missing" in low:
        missing_cells = int(df.isna().sum().sum())
        missing_rows = int(df.isna().any(axis=1).sum())
        if "which column" in low or "columns" in low:
            items = df.isna().sum().sort_values(ascending=False)
            items = items[items > 0].head(5)
            if len(items):
                return "Highest missing-value columns: " + ", ".join(
                    f"{idx} ({int(value):,})" for idx, value in items.items()
                ) + "."
        return f"There are {missing_cells:,} missing cells across {missing_rows:,} rows."

    # Metric questions.
    if metric:
        values = numeric_values(metric)
        if len(values):
            if any(k in low for k in ["total", "sum", "revenue", "sales", "amount", "profit"]):
                return f"Total {metric} is {format_number(values.sum())} across {len(values):,} valid records."
            if any(k in low for k in ["average", "avg", "mean"]):
                return f"Average {metric} is {format_number(values.mean())} across {len(values):,} valid values."
            if any(k in low for k in ["highest", "maximum", "max", "largest"]):
                return f"Highest {metric} is {format_number(values.max())}."
            if any(k in low for k in ["lowest", "minimum", "min", "smallest"]):
                return f"Lowest {metric} is {format_number(values.min())}."

    if metric2 and any(k in low for k in ["quantity", "qty", "units"]):
        values = numeric_values(metric2)
        if len(values):
            return f"Total {metric2} is {format_number(values.sum())} across {len(values):,} valid records."

    # Category and location questions.
    if category and any(k in low for k in ["top", "highest", "best", "most"]):
        rows = aggregate_category(df, category, metric)
        if rows:
            return f"Top {category} by {metric or 'records'} is {rows[0]['label']} with {rows[0]['display']}."
    if category and any(k in low for k in ["categories", "unique category", "different category"]):
        return f"There are {int(df[category].nunique(dropna=True)):,} unique {category} values."

    if location and any(k in low for k in ["state", "location", "region", "city", "country"]):
        rows = aggregate_category(df, location, metric)
        if rows:
            return f"Top {location} by {metric or 'records'} is {rows[0]['label']} with {rows[0]['display']}."

    # Date/trend questions.
    if date_column and any(k in low for k in ["trend", "over time", "month", "date"]):
        trend = aggregate_trend(df, date_column, metric, metric2)
        if len(trend) >= 2:
            first = trend[0]
            last = trend[-1]
            return f"The {metric or 'primary metric'} trend runs from {first.get('label')} to {last.get('label')}. Review the dashboard trend chart for the period-by-period values."
        return "There are not enough valid time periods to describe a reliable trend."

    if any(k in low for k in ["column names", "what columns", "columns are there"]):
        return "Analyzed columns: " + ", ".join(str(c) for c in df.columns) + "."

    return (
        f"I can analyze {len(df):,} rows and {len(df.columns):,} columns. "
        f"Try: 'What is the total {metric or 'amount'}?', 'What is the average?', "
        f"'Which {category or 'category'} is highest?', 'Which location is highest?', "
        "'How many missing values?', or 'Are there duplicates?'"
    )


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------


@app.get("/")
def home():
    return {
        "app": "InsightAI",
        "message": "InsightAI backend is running successfully.",
        "status": "online",
        "version": "6.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "InsightAI API",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    allowed_extensions = (".xlsx", ".xls", ".csv")
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only .xlsx, .xls and .csv files are supported.",
        )

    try:
        file_data = await file.read()
        if not file_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_data))
        else:
            df = pd.read_excel(BytesIO(file_data))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file contains no data.",
            )

        # Clean column names.
        df.columns = [str(column).strip() for column in df.columns]

        # Remove completely empty columns.
        df = df.dropna(axis=1, how="all")

        # Remove only obvious generated CSV/Excel index columns.
        generated_index_columns = [
            str(column)
            for column in df.columns
            if is_generated_index_column(column, df[column])
        ]
        if generated_index_columns:
            df = df.drop(columns=generated_index_columns)

        if df.empty or len(df.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="No usable data columns were found in the file.",
            )

        total_rows = int(len(df))
        total_columns = int(len(df.columns))
        columns = [str(column) for column in df.columns]

        missing_values = int(df.isna().sum().sum())
        rows_with_missing = int(df.isna().any(axis=1).sum())
        duplicate_rows = int(df.duplicated().sum())

        # -------------------------------------------------
        # COLUMN DETECTION
        # -------------------------------------------------

        identifier_columns = []
        numeric_columns = []
        text_columns = []
        date_columns = []

        for column in df.columns:
            series = df[column]

            # Detect dates before generic high-cardinality ID detection.
            # A real date column can have almost one unique value per row,
            # which otherwise makes a generic ID heuristic misclassify it.
            if detect_date_column(df, column):
                date_columns.append(str(column))
                continue

            if looks_like_id(column, series):
                identifier_columns.append(str(column))
                continue

            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(str(column))
                continue

            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                text_columns.append(str(column))

        # Category columns should have useful cardinality.
        category_columns = []
        for column in text_columns:
            unique_count = int(df[column].nunique(dropna=True))
            if unique_count <= max(100, int(total_rows * 0.5)):
                category_columns.append(column)

        metric = choose_metric(numeric_columns, df=df, identifier_columns=identifier_columns, date_columns=date_columns)
        metric2 = choose_secondary_metric(numeric_columns, metric, df=df, identifier_columns=identifier_columns, date_columns=date_columns)
        category = choose_category(category_columns, df=df)
        region = choose_region(category_columns, category)
        date_column = date_columns[0] if date_columns else None

        date_parse_quality = None
        if date_column:
            raw_date = df[date_column].dropna()
            parsed_date = parse_dates(raw_date)
            date_parse_quality = round(
                float(parsed_date.notna().mean() * 100) if len(raw_date) else 0,
                1,
            )

        location_column, location_type = choose_location_column(
            text_columns,
            category_columns,
        )

        # Do not use the same column as both main category and location chart.
        if location_column == category:
            alternative_location = choose_region(category_columns, category)
            if alternative_location:
                location_column = alternative_location
                location_type = "location"
            else:
                location_column = None
                location_type = None

        # -------------------------------------------------
        # COLUMN DETAILS
        # -------------------------------------------------

        column_details = []
        for column in df.columns:
            series = df[column]
            name = str(column)

            if name in identifier_columns:
                detected_type = "Identifier"
            elif name in date_columns:
                detected_type = "Date"
            elif name in numeric_columns:
                detected_type = "Numeric"
            else:
                detected_type = "Text"

            column_details.append(
                {
                    "name": name,
                    "dtype": str(series.dtype),
                    "detected_type": detected_type,
                    "missing": int(series.isna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                }
            )

        # -------------------------------------------------
        # CHART DATA
        # -------------------------------------------------

        category_data = aggregate_category(df, category, metric)
        trend_data = aggregate_trend(df, date_column, metric)

        # Separate secondary trend so Quantity does not share Amount's scale.
        secondary_trend = aggregate_trend(df, date_column, metric2)
        distribution = calculate_distribution(category_data)

        metric_total = None
        if metric:
            metric_total = float(
                pd.to_numeric(df[metric], errors="coerce").sum()
            )

        secondary_total = None
        if metric2:
            secondary_total = float(
                pd.to_numeric(df[metric2], errors="coerce").sum()
            )

        # Never fabricate a time axis. If no reliable date column exists, the
        # dashboard can still show category/location analysis, but trend insight
        # is omitted rather than presenting arbitrary row groups as time.

        # Location/state chart data.
        location_data = aggregate_location(
            df,
            location_column,
            metric,
            location_type=location_type or "location",
        )

        # Advanced, dataset-agnostic analytics. Boolean/flag fields such as B2B
        # are not business measures, so keep them out of KPI/statistical analysis.
        flag_columns = [
            str(column) for column in numeric_columns
            if is_flag_column(column, df[column])
        ]
        usable_numeric_columns = [
            column for column in numeric_columns
            if column not in flag_columns
        ]
        missing_by_column = build_missing_by_column(df)
        outlier_summary = build_outlier_summary(df, usable_numeric_columns)
        correlation_summary = build_correlations(df, usable_numeric_columns)
        anomaly_summary = build_anomalies(trend_data, metric)
        dynamic_kpis = build_dynamic_kpis(df, metric, metric2, category, location_column)
        stat_summary = build_stat_summary(df, usable_numeric_columns)
        advanced_insights = build_advanced_insights(
            metric,
            metric2,
            outlier_summary,
            correlation_summary,
            anomaly_summary,
            missing_by_column,
        )

        # -------------------------------------------------
        # DATASET IDENTITY
        # -------------------------------------------------

        dataset_identity = detect_dataset_identity(df, columns)

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        total_cells = total_rows * total_columns
        completeness = (
            (total_cells - missing_values) / total_cells * 100
            if total_cells
            else 100
        )
        missing_percent = (
            missing_values / total_cells * 100
            if total_cells
            else 0
        )
        duplicate_percent = (
            duplicate_rows / total_rows * 100
            if total_rows
            else 0
        )

        # -------------------------------------------------
        # PREVIEW
        # -------------------------------------------------

        preview = clean_records(df.head(100).copy())

        # -------------------------------------------------
        # INSIGHTS
        # -------------------------------------------------

        insights = (build_insights(
            df=df,
            metric=metric,
            category=category,
            region=region,
            location_column=location_column,
            location_type=location_type or "location",
            date_column=date_column,
            category_data=category_data,
            trend_data=trend_data,
            metric_total=metric_total,
            missing_values=missing_values,
            missing_percent=missing_percent,
            duplicate_rows=duplicate_rows,
            rows_with_missing=rows_with_missing,
        ) + advanced_insights)[:10]

        analysis_id = str(uuid.uuid4())
        ANALYSIS_SESSIONS[analysis_id] = {
            "df": df.copy(),
            "filename": filename,
            "metric": metric,
            "metric2": metric2,
            "category": category,
            "region": region,
            "date_column": date_column,
            "date_columns": date_columns,
            "location_column": location_column,
            "location_type": location_type,
            "identifier_columns": identifier_columns,
            "flag_columns": flag_columns,
            "numeric_columns": numeric_columns,
            "generated_columns": generated_index_columns,
            "created_at": datetime.utcnow().isoformat(),
        }
        _session_cleanup()

        return {
            "success": True,
            "analysis_id": analysis_id,
            "filename": filename,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": columns,
            "removed_generated_columns": generated_index_columns,

            "missing_values": missing_values,
            "rows_with_missing": rows_with_missing,
            "duplicate_rows": duplicate_rows,

            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "identifier_columns": identifier_columns,
            "flag_columns": flag_columns,
            "date_columns": date_columns,

            "primary_metric": metric,
            "secondary_metric": metric2,
            "primary_category": category,
            "primary_date": date_column,
            "date_parse_quality": date_parse_quality,
            "location_column": location_column,
            "location_type": location_type,

            "metric_total": clean_value(metric_total),
            "metric_total_display": (
                format_number(metric_total)
                if metric_total is not None
                else None
            ),
            "secondary_total": clean_value(secondary_total),
            "secondary_total_display": (
                format_number(secondary_total)
                if secondary_total is not None
                else None
            ),

            "dataset_identity": dataset_identity,
            "column_details": column_details,
            "preview": preview,

            "charts": {
                "trend": trend_data,
                "secondary_trend": secondary_trend,
                "category": category_data,
                "distribution": distribution,
                "location": location_data,
            },

            "quality": {
                "completeness": round(completeness, 1),
                "missing_cells": missing_values,
                "missing_percent": round(missing_percent, 1),
                "rows_with_missing": rows_with_missing,
                "duplicate_rows": duplicate_rows,
                "duplicate_percent": round(duplicate_percent, 1),
                "columns_with_missing": len(missing_by_column),
                "outlier_columns": len(outlier_summary),
            },

            "analytics": {
                "kpis": dynamic_kpis,
                "missing_by_column": missing_by_column,
                "outliers": outlier_summary,
                "correlations": correlation_summary,
                "anomalies": anomaly_summary,
                "stat_summary": stat_summary,
            },

            "insights": insights,
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"ERROR while processing {filename}:", error)
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not analyze the file. "
                f"Reason: {str(error)}"
            ),
        )


@app.post("/filter")
async def filter_analysis(payload: dict):
    analysis_id = payload.get("analysis_id")
    session = _get_session(analysis_id)
    filtered_df = _apply_filters(session["df"], payload.get("filters"))
    result = _build_analysis_payload(filtered_df, session)
    result.update({
        "success": True, "analysis_id": analysis_id, "filename": session["filename"],
        "primary_metric": session["metric"], "secondary_metric": session["metric2"],
        "primary_category": session["category"], "primary_date": session["date_column"],
        "location_column": session["location_column"], "location_type": session["location_type"],
        "identifier_columns": session["identifier_columns"], "flag_columns": session["flag_columns"],
        "removed_generated_columns": session["generated_columns"],
        "dataset_identity": detect_dataset_identity(filtered_df, [str(c) for c in filtered_df.columns]),
        "column_details": [{"name":str(c),"dtype":str(filtered_df[c].dtype),"detected_type":("Identifier" if str(c) in session["identifier_columns"] else ("Date" if session["date_column"] and str(c) == session["date_column"] else ("Numeric" if str(c) in session["numeric_columns"] else "Text"))),"missing":int(filtered_df[c].isna().sum()),"unique":int(filtered_df[c].nunique(dropna=True))} for c in filtered_df.columns],
        "preview": clean_records(filtered_df.head(100)),
    })
    return result


@app.post("/chat")
async def chat(payload: dict):
    session = _get_session(payload.get("analysis_id"))
    return {"success": True, "answer": _chat_answer(session, payload.get("question"))}


@app.post("/forecast")
async def forecast(payload: dict):
    session = _get_session(payload.get("analysis_id"))
    periods = max(1, min(int(payload.get("periods", 6)), 12))
    metric = _resolve_forecast_metric(session)
    date_column = _resolve_forecast_date(session)
    result = _forecast(session["df"], metric, date_column, periods)
    return {
        "success": True,
        "analysis_id": payload.get("analysis_id"),
        **result,
    }


@app.post("/report")
async def report(payload: dict):
    session = _get_session(payload.get("analysis_id"))
    result = _build_analysis_payload(session["df"], session)
    fmt = str(payload.get("format", "csv")).lower()
    if fmt == "csv":
        rows = clean_records(session["df"].head(100))
        if rows:
            out = pd.DataFrame(rows).to_csv(index=False)
        else:
            out = "InsightAI report\n"
        return {"success": True, "format": "csv", "filename": "insightai_report.csv", "content": out}
    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            y = height - 50
            pdf.setFont("Helvetica-Bold", 16); pdf.drawString(40, y, "InsightAI Data Report"); y -= 28
            pdf.setFont("Helvetica", 10)
            lines = [f"File: {session['filename']}", f"Rows: {len(session['df']):,}", f"Columns: {len(session['df'].columns)}", f"Primary metric: {session['metric'] or 'Not detected'}", f"Category: {session['category'] or 'Not detected'}", f"Date: {session['date_column'] or 'Not detected'}", "", "Insights:"] + [f"- {x['title']}: {x['text']}" for x in result.get('insights', [])]
            for line in lines:
                if y < 50: pdf.showPage(); y = height - 50; pdf.setFont("Helvetica", 10)
                pdf.drawString(40, y, line[:120]); y -= 16
            pdf.save(); buffer.seek(0)
            import base64
            return {"success": True, "format":"pdf", "filename":"insightai_report.pdf", "content_base64":base64.b64encode(buffer.read()).decode()}
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF export requires reportlab on the backend.")
    raise HTTPException(status_code=400, detail="Supported report formats: csv, pdf")
