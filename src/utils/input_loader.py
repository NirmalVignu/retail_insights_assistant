"""
Input Loader Utility
Supports CSV, Excel, JSON, and text summary files for retail insights ingestion.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Any, Optional
import json
import re
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _clean_table_name(name: str) -> str:
    """Normalize file names into SQL-safe table names."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "loaded_table"


def infer_table_name(filename: str) -> str:
    """Infer table name from file name."""
    return _clean_table_name(Path(filename).stem)


def load_dataframe_from_bytes(file_name: str, file_bytes: bytes, auto_clean: bool = True) -> pd.DataFrame:
    """
    Load a dataframe from uploaded file bytes based on extension.
    
    Args:
        file_name: Name of the file
        file_bytes: File content as bytes
        auto_clean: If True, automatically clean the dataframe
    
    Returns:
        Loaded (and optionally cleaned) DataFrame
    """
    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(BytesIO(file_bytes))
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(BytesIO(file_bytes))
    elif suffix == ".json":
        text = file_bytes.decode("utf-8", errors="ignore")
        payload = json.loads(text)
        if isinstance(payload, list):
            df = pd.json_normalize(payload)
        elif isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                df = pd.json_normalize(payload["data"])
            else:
                df = pd.json_normalize([payload])
        else:
            raise ValueError("Unsupported JSON structure")
    elif suffix == ".txt":
        df = parse_text_summary(file_bytes.decode("utf-8", errors="ignore"))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    # Auto-clean if enabled
    if auto_clean:
        df = clean_dataframe(df, strict=False)
    
    return df


def parse_text_summary(text: str) -> pd.DataFrame:
    """Parse a simple summary text into key/value rows.

    Supported line formats:
    - key: value
    - key = value
    - plain text lines (captured as notes)
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            rows.append({"metric": key.strip(), "value": value.strip(), "source": "text"})
        elif "=" in line:
            key, value = line.split("=", 1)
            rows.append({"metric": key.strip(), "value": value.strip(), "source": "text"})
        else:
            rows.append({"metric": "note", "value": line, "source": "text"})

    if not rows:
        rows.append({"metric": "note", "value": "No parsable content found", "source": "text"})

    return pd.DataFrame(rows)


def clean_dataframe(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Clean DataFrame by removing irrelevant rows and fixing data quality issues.
    
    Args:
        df: Input DataFrame to clean
        strict: If True, apply stricter cleaning rules (e.g., remove rows with >80% nulls)
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    original_rows = len(df)
    logger.info(f"Starting data cleaning: {original_rows} rows")
    
    # 1. Trim whitespace from all text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        # Replace empty strings with NaN
        df[col] = df[col].replace('', np.nan)
        df[col] = df[col].replace('nan', np.nan)
    
    # 2. Remove completely empty rows
    df = df.dropna(how='all')
    logger.info(f"Removed {original_rows - len(df)} completely empty rows")
    
    # 3. Remove duplicate header rows (where row values match column names)
    header_mask = df.apply(
        lambda row: row.astype(str).str.lower().isin(df.columns.str.lower()).sum() >= len(df.columns) * 0.7,
        axis=1
    )
    rows_before = len(df)
    df = df[~header_mask]
    if rows_before > len(df):
        logger.info(f"Removed {rows_before - len(df)} duplicate header rows")
    
    # 4. Remove summary rows (Total, Subtotal, Grand Total, etc.)
    summary_keywords = [
        'total', 'subtotal', 'grand total', 'sum', 'average', 'avg',
        'overall', 'summary', 'grand sum', 'net total', 'gross total'
    ]
    rows_before = len(df)
    for col in text_cols:
        if col in df.columns:
            mask = df[col].astype(str).str.lower().str.contains(
                '|'.join(summary_keywords), na=False, regex=True
            )
            df = df[~mask]
    if rows_before > len(df):
        logger.info(f"Removed {rows_before - len(df)} summary/total rows")
    
    # 5. Fix numeric columns stored as text
    # Detect columns that look numeric but are stored as objects
    for col in text_cols:
        if col in df.columns:
            # Check if >50% of non-null values can be converted to numbers
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                if numeric_count / len(sample) > 0.5:
                    # Convert column to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted column '{col}' to numeric type")
    
    # 6. STRICT MODE: Remove rows with mostly null values (>80% null)
    if strict:
        null_threshold = 0.8
        null_ratio = df.isnull().sum(axis=1) / len(df.columns)
        rows_before = len(df)
        df = df[null_ratio <= null_threshold]
        if rows_before > len(df):
            logger.info(f"[STRICT] Removed {rows_before - len(df)} rows with >{null_threshold*100}% null values")
    
    # 7. Reset index
    df = df.reset_index(drop=True)
    
    cleaned_rows = len(df)
    logger.info(f"Cleaning complete: {original_rows} → {cleaned_rows} rows ({original_rows - cleaned_rows} removed)")
    
    return df

