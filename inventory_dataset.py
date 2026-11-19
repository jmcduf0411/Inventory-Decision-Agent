"""
Supply Chain Analytics: Inventory Dataset Processing & Validation
==================================================================
Handles CSV inventory dataset upload, schema validation, data cleaning,
and basic descriptive analysis.

Required schema:
- Product (str)
- Category (str)
- Units (numeric, non-negative)
- Unit Cost (numeric, non-negative)

Calculates:
- Inventory Value = Units * Unit Cost
- Descriptive analysis: total products, total units, total inventory value,
  and category breakdowns.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
import io
import pandas as pd


REQUIRED_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]


@dataclass
class InventoryDatasetSummary:
    num_products: int
    total_units: float
    total_inventory_value: float
    num_categories: int
    avg_unit_cost: float
    warnings: List[str] = field(default_factory=list)
    raw_row_count: int = 0
    clean_row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names by stripping whitespace, UTF-8 BOM,
    and mapping case-insensitive variants to canonical required column names.
    """
    df = df.copy()
    cleaned_cols = {}
    for col in df.columns:
        norm = str(col).strip().replace("\ufeff", "")
        # Map case-insensitive canonical names
        low = norm.lower()
        if low == "product":
            cleaned_cols[col] = "Product"
        elif low == "category":
            cleaned_cols[col] = "Category"
        elif low in ["units", "quantity", "inventory_units", "stock_units"]:
            cleaned_cols[col] = "Units"
        elif low in ["unit cost", "unit_cost", "unitcost", "cost", "unit price"]:
            cleaned_cols[col] = "Unit Cost"
        else:
            cleaned_cols[col] = norm
    return df.rename(columns=cleaned_cols)


def validate_inventory_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the DataFrame contains all four required columns:
    Product, Category, Units, Unit Cost.
    
    Returns:
        (is_valid, missing_columns)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, list(REQUIRED_COLUMNS)

    normalized_df = normalize_column_names(df)
    existing_cols = list(normalized_df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in existing_cols]

    return (len(missing) == 0, missing)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Cleans strings with currency symbols ($), commas, and spaces, then casts to float.
    Invalid values become NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def process_inventory_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, InventoryDatasetSummary]:
    """
    Validates, cleans, and computes inventory metrics for an uploaded DataFrame.
    
    Raises:
        ValueError: If required columns are missing or the dataset is empty.
        
    Returns:
        (clean_df, summary_object)
    """
    if df is None or df.empty:
        raise ValueError("The uploaded CSV file is empty. Please provide a file with inventory records.")

    normalized_df = normalize_column_names(df)
    is_valid, missing = validate_inventory_dataset(normalized_df)
    if not is_valid:
        raise ValueError(
            f"Uploaded dataset is missing required column(s): {', '.join(missing)}. "
            f"The CSV must contain: {', '.join(REQUIRED_COLUMNS)}."
        )

    raw_row_count = len(normalized_df)
    warnings: List[str] = []

    clean_df = normalized_df.copy()

    # Clean product & category strings
    clean_df["Product"] = clean_df["Product"].astype(str).str.strip()
    clean_df["Category"] = clean_df["Category"].astype(str).str.strip()

    # Drop rows where Product is completely missing or empty
    empty_products = clean_df["Product"].isin(["", "nan", "None", "null"])
    if empty_products.any():
        dropped_count = empty_products.sum()
        clean_df = clean_df[~empty_products].copy()
        warnings.append(f"Dropped {dropped_count} row(s) with missing or empty Product names.")

    # Clean numeric columns
    clean_df["Units"] = clean_numeric_series(clean_df["Units"])
    clean_df["Unit Cost"] = clean_numeric_series(clean_df["Unit Cost"])

    # Identify invalid numerics (NaN or negative)
    invalid_units = clean_df["Units"].isna() | (clean_df["Units"] < 0)
    invalid_cost = clean_df["Unit Cost"].isna() | (clean_df["Unit Cost"] < 0)
    invalid_mask = invalid_units | invalid_cost

    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        warnings.append(
            f"Removed {invalid_count} row(s) containing missing, non-numeric, or negative Units / Unit Cost values."
        )
        clean_df = clean_df[~invalid_mask].copy()

    if clean_df.empty:
        raise ValueError(
            "No valid inventory rows remain after removing missing or invalid numeric entries. "
            "Please ensure Units and Unit Cost contain non-negative numbers."
        )

    # Calculate Inventory Value = Units * Unit Cost
    clean_df["Inventory Value"] = clean_df["Units"] * clean_df["Unit Cost"]

    # Compute descriptive summary metrics
    total_units = float(clean_df["Units"].sum())
    total_inventory_value = float(clean_df["Inventory Value"].sum())
    num_products = int(clean_df["Product"].nunique())
    num_categories = int(clean_df["Category"].nunique())
    avg_unit_cost = float(clean_df["Unit Cost"].mean()) if len(clean_df) > 0 else 0.0

    summary = InventoryDatasetSummary(
        num_products=num_products,
        total_units=total_units,
        total_inventory_value=total_inventory_value,
        num_categories=num_categories,
        avg_unit_cost=avg_unit_cost,
        warnings=warnings,
        raw_row_count=raw_row_count,
        clean_row_count=len(clean_df),
    )

    return clean_df, summary


def load_inventory_csv_from_bytes(file_bytes_or_buffer: Any) -> Tuple[pd.DataFrame, InventoryDatasetSummary]:
    """
    Helper to read bytes / uploaded file buffer directly from Streamlit's file_uploader.
    """
    try:
        raw_df = pd.read_csv(file_bytes_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")

    return process_inventory_dataset(raw_df)
