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

import pandas as pd


def _clean_table_name(name: str) -> str:
    """Normalize file names into SQL-safe table names."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "loaded_table"


def infer_table_name(filename: str) -> str:
    """Infer table name from file name."""
    return _clean_table_name(Path(filename).stem)


def load_dataframe_from_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Load a dataframe from uploaded file bytes based on extension."""
    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(BytesIO(file_bytes))

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(file_bytes))

    if suffix == ".json":
        text = file_bytes.decode("utf-8", errors="ignore")
        payload = json.loads(text)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return pd.json_normalize(payload["data"])
            return pd.json_normalize([payload])
        raise ValueError("Unsupported JSON structure")

    if suffix == ".txt":
        return parse_text_summary(file_bytes.decode("utf-8", errors="ignore"))

    raise ValueError(f"Unsupported file type: {suffix}")


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
