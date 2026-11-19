"""
Supply Chain Analytics: Inventory Dataset Processing & Multicriteria Classification
====================================================================================
Handles CSV inventory dataset upload, schema validation, data cleaning,
basic descriptive analysis, and BAN 517 multicriteria inventory classification.

Supported Schemas:
1. Classification Schema (Multicriteria using Sales & Hits):
   - Item Number
   - Item Name
   - UoM
   - Sales ($)
   - Hits

2. Valuation / Descriptive Schema:
   - Product
   - Category
   - Units
   - Unit Cost

BAN 517 Multicriteria Methodology:
----------------------------------
Step 1: Classify Sales ($) and Hits independently using Pareto cumulative thresholds:
  - Class A (Score 4): 0–60% cumulative contribution
  - Class B (Score 3): 61–80% cumulative contribution
  - Class C (Score 2): 81–90% cumulative contribution
  - Class D (Score 1): 91–100% cumulative contribution

Step 2: Calculate Combined Weighted Score:
  Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
  (e.g., Sales A [4] and Hits B [3] -> 0.65*4 + 0.35*3 = 3.65)

Step 3: Assign Final Inventory Class:
  - Class A: 3.25 through 4.00
  - Class B: 2.50 up to but not including 3.25
  - Class C: 1.75 up to but not including 2.50
  - Class D: Below 1.75
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
import io
import pandas as pd
import numpy as np

__all__ = [
    "REQUIRED_COLUMNS",
    "CLASSIFICATION_REQUIRED_COLUMNS",
    "TIER_SCORES",
    "CLASS_STRATEGIES",
    "InventoryDatasetSummary",
    "MulticriteriaClassSummary",
    "MulticriteriaClassificationSummary",
    "normalize_column_names",
    "validate_inventory_dataset",
    "validate_classification_dataset",
    "detect_dataset_schema",
    "clean_numeric_series",
    "assign_pareto_tier",
    "assign_final_class_from_score",
    "classify_inventory_multicriteria",
    "process_inventory_dataset",
    "load_inventory_csv_from_bytes",
]

# Canonical required columns for both supported CSV schemas
REQUIRED_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]
CLASSIFICATION_REQUIRED_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]

# Score mapping for Pareto tiers
TIER_SCORES: Dict[str, int] = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}

# Final class managerial descriptions
CLASS_STRATEGIES: Dict[str, str] = {
    "A": "Class A (Weighted Score 3.25 - 4.00): Critical revenue & velocity drivers. Requires continuous review, tight control, high service levels, and prime warehouse slotting.",
    "B": "Class B (Weighted Score 2.50 - 3.24): Moderate inventory impact. Managed via standard periodic/continuous replenishment (EOQ/ROP) and standard warehouse storage.",
    "C": "Class C (Weighted Score 1.75 - 2.49): Low revenue and activity. Simple periodic review, bulk ordering, lower safety stocks, and higher-density storage.",
    "D": "Class D (Weighted Score < 1.75): Minimal contribution / slow-movers. Review for SKU rationalization, liquidation, obsolete write-off, or vendor consignment.",
}


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


@dataclass
class MulticriteriaClassSummary:
    class_name: str
    item_count: int
    pct_items: float
    total_sales: float
    pct_sales: float
    total_hits: float
    pct_hits: float
    strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MulticriteriaClassificationSummary:
    total_items: int
    total_sales: float
    total_hits: float
    class_breakdown: Dict[str, MulticriteriaClassSummary]
    cross_matrix: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    raw_row_count: int = 0
    clean_row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_sales": self.total_sales,
            "total_hits": self.total_hits,
            "class_breakdown": {k: v.to_dict() for k, v in self.class_breakdown.items()},
            "warnings": self.warnings,
            "raw_row_count": self.raw_row_count,
            "clean_row_count": self.clean_row_count,
        }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names by stripping whitespace, UTF-8 BOM,
    and mapping case-insensitive synonyms to canonical column names
    for both classification and valuation schemas.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    df = df.copy()
    cleaned_cols = {}
    for col in df.columns:
        norm = str(col).strip().replace("\ufeff", "")
        low = norm.lower()

        # Valuation schema synonyms
        if low in ["product", "product_name", "product name"]:
            cleaned_cols[col] = "Product"
        elif low in ["category", "prod_category", "item category"]:
            cleaned_cols[col] = "Category"
        elif low in ["units", "quantity", "inventory_units", "stock_units", "qty", "on hand units"]:
            cleaned_cols[col] = "Units"
        elif low in ["unit cost", "unit_cost", "unitcost", "cost", "unit price", "price"]:
            cleaned_cols[col] = "Unit Cost"

        # Classification schema synonyms
        elif low in ["item number", "item no", "item #", "item_number", "item", "item code", "sku", "part number", "part #"]:
            cleaned_cols[col] = "Item Number"
        elif low in ["item name", "item_name", "item description", "description", "product description", "title"]:
            cleaned_cols[col] = "Item Name"
        elif low in ["uom", "unit of measure", "unit of measurement", "measure", "unit"]:
            cleaned_cols[col] = "UoM"
        elif low in ["sales ($)", "sales", "sales$", "annual sales", "revenue", "sales amount", "sales_usd", "total sales"]:
            cleaned_cols[col] = "Sales ($)"
        elif low in ["hits", "hit", "picks", "order lines", "transactions", "order frequency", "hit count", "pick count"]:
            cleaned_cols[col] = "Hits"
        else:
            cleaned_cols[col] = norm

    return df.rename(columns=cleaned_cols)


def validate_inventory_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the DataFrame contains all four required valuation columns:
    Product, Category, Units, Unit Cost.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, list(REQUIRED_COLUMNS)

    normalized_df = normalize_column_names(df)
    existing_cols = list(normalized_df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in existing_cols]

    return (len(missing) == 0, missing)


def validate_classification_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the DataFrame contains all five required classification columns:
    Item Number, Item Name, UoM, Sales ($), Hits.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, list(CLASSIFICATION_REQUIRED_COLUMNS)

    normalized_df = normalize_column_names(df)
    existing_cols = list(normalized_df.columns)
    missing = [c for c in CLASSIFICATION_REQUIRED_COLUMNS if c not in existing_cols]

    return (len(missing) == 0, missing)


def detect_dataset_schema(df: pd.DataFrame) -> str:
    """
    Detects which inventory schema the uploaded DataFrame represents:
    - 'classification': contains Item Number, Item Name, UoM, Sales ($), Hits
    - 'valuation': contains Product, Category, Units, Unit Cost
    - 'unknown': does not match either schema
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return "unknown"

    normalized_df = normalize_column_names(df)
    cols = set(normalized_df.columns)

    if set(CLASSIFICATION_REQUIRED_COLUMNS).issubset(cols):
        return "classification"
    if set(REQUIRED_COLUMNS).issubset(cols):
        return "valuation"
    return "unknown"


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Cleans strings with currency symbols ($), commas, and spaces, then casts to float.
    Invalid or unparseable values become NaN.
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


def assign_pareto_tier(
    series: pd.Series,
    item_numbers: Optional[pd.Series] = None,
    a_cutoff: float = 0.60,
    b_cutoff: float = 0.80,
    c_cutoff: float = 0.90,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Step 1: Classifies a single criterion (Sales or Hits) independently according
    to BAN 517 Pareto classification thresholds:
    - Class A (Score 4): 0–60% cumulative contribution
    - Class B (Score 3): 61–80% cumulative contribution
    - Class C (Score 2): 81–90% cumulative contribution
    - Class D (Score 1): 91–100% cumulative contribution

    Tie-breaking & boundary handling:
    - Items are sorted descending by value. Tied values are stably resolved by secondary
      sort on Item Number (or original index).
    - Cumulative contribution is evaluated sequentially.
    - If an item's preceding cumulative sum is < 0.60 (or it is the top item), it is placed in A.
    - If preceding cumulative sum is < 0.80, it is placed in B.
    - If preceding cumulative sum is < 0.90, it is placed in C.
    - Remaining items (or items starting at >= 0.90) are placed in D.
    - This boundary-crossing convention guarantees that boundary-spanning items contributing
      to a tranche are rewarded to that tranche, and Class A is never empty if positive values exist.

    Returns:
        (cum_pct_series, class_tier_series, score_series) indexed matching the original series.
    """
    df_temp = pd.DataFrame({"val": series}, index=series.index)
    if item_numbers is not None:
        df_temp["item_key"] = item_numbers
        df_sorted = df_temp.sort_values(by=["val", "item_key"], ascending=[False, True])
    else:
        df_sorted = df_temp.sort_values(by="val", ascending=False)

    total = float(df_sorted["val"].clip(lower=0.0).sum())
    if total <= 0:
        cum_pct = pd.Series(0.0, index=series.index)
        classes = pd.Series("D", index=series.index)
        scores = pd.Series(1, index=series.index)
        return cum_pct, classes, scores

    pos_vals = df_sorted["val"].clip(lower=0.0)
    cumsum = pos_vals.cumsum()
    cum_pct_sorted = cumsum / total
    prev_cum = cum_pct_sorted.shift(1, fill_value=0.0)

    classes_sorted = []
    scores_sorted = []
    for val, p_cum, c_cum in zip(df_sorted["val"], prev_cum, cum_pct_sorted):
        if val <= 0:
            # Non-contributing / zero items go to Class D (Score 1)
            classes_sorted.append("D")
            scores_sorted.append(1)
        elif p_cum < a_cutoff:
            classes_sorted.append("A")
            scores_sorted.append(4)
        elif p_cum < b_cutoff:
            classes_sorted.append("B")
            scores_sorted.append(3)
        elif p_cum < c_cutoff:
            classes_sorted.append("C")
            scores_sorted.append(2)
        else:
            classes_sorted.append("D")
            scores_sorted.append(1)

    s_classes = pd.Series(classes_sorted, index=df_sorted.index).loc[series.index]
    s_scores = pd.Series(scores_sorted, index=df_sorted.index).loc[series.index]
    s_cum_pct = cum_pct_sorted.loc[series.index]

    return s_cum_pct, s_classes, s_scores


def assign_final_class_from_score(weighted_score: float) -> str:
    """
    Step 3: Assigns the final inventory class based on the combined Weighted Score:
    - Class A: 3.25 through 4.00
    - Class B: 2.50 up to but not including 3.25
    - Class C: 1.75 up to but not including 2.50
    - Class D: Below 1.75
    """
    score = round(float(weighted_score), 4)
    if score >= 3.25:
        return "A"
    elif score >= 2.50:
        return "B"
    elif score >= 1.75:
        return "C"
    else:
        return "D"


def classify_inventory_multicriteria(
    df: pd.DataFrame,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> Tuple[pd.DataFrame, MulticriteriaClassificationSummary]:
    """
    Executes BAN 517 multicriteria inventory classification into A, B, C, and D classes
    using Sales ($) and Hits.

    Methodology:
    Step 1: Classify Sales ($) and Hits separately using Pareto thresholds (0-60% A, 61-80% B, 81-90% C, 91-100% D)
            Assign scores: A=4, B=3, C=2, D=1.
    Step 2: Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score.
    Step 3: Final Inventory Class:
            - Class A: 3.25 through 4.00
            - Class B: 2.50 up to but not including 3.25
            - Class C: 1.75 up to but not including 2.50
            - Class D: Below 1.75

    Enriched DataFrame columns:
    - Item Number, Item Name, UoM, Sales ($), Hits
    - Sales cumulative percentage, Sales Class, Sales Score
    - Hits cumulative percentage, Hits Class, Hits Score
    - Weighted Score, Final Inventory Class

    Returns:
        (clean_df, summary_object)
    """
    if df is None or df.empty:
        raise ValueError("The uploaded CSV file is empty. Please provide a file with inventory records.")

    normalized_df = normalize_column_names(df)
    is_valid, missing = validate_classification_dataset(normalized_df)
    if not is_valid:
        raise ValueError(
            f"Uploaded dataset is missing required classification column(s): {', '.join(missing)}. "
            f"Classification CSV must contain: {', '.join(CLASSIFICATION_REQUIRED_COLUMNS)}."
        )

    raw_row_count = len(normalized_df)
    warnings: List[str] = []
    clean_df = normalized_df.copy()

    # Clean text columns
    clean_df["Item Number"] = clean_df["Item Number"].astype(str).str.strip()
    clean_df["Item Name"] = clean_df["Item Name"].astype(str).str.strip()
    clean_df["UoM"] = clean_df["UoM"].astype(str).str.strip()

    # Drop rows where Item Number is missing or empty
    empty_items = clean_df["Item Number"].isin(["", "nan", "None", "null"])
    if empty_items.any():
        dropped_count = int(empty_items.sum())
        clean_df = clean_df[~empty_items].copy()
        warnings.append(f"Dropped {dropped_count} row(s) with missing or empty Item Number.")

    # Clean numeric columns: Sales ($) and Hits
    clean_df["Sales ($)"] = clean_numeric_series(clean_df["Sales ($)"])
    clean_df["Hits"] = clean_numeric_series(clean_df["Hits"])

    # Identify invalid numerics (NaN or negative)
    invalid_sales = clean_df["Sales ($)"].isna() | (clean_df["Sales ($)"] < 0)
    invalid_hits = clean_df["Hits"].isna() | (clean_df["Hits"] < 0)
    invalid_mask = invalid_sales | invalid_hits

    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        warnings.append(
            f"Removed {invalid_count} row(s) containing missing, non-numeric, or negative Sales ($) or Hits values."
        )
        clean_df = clean_df[~invalid_mask].copy()

    if clean_df.empty:
        raise ValueError(
            "No valid inventory rows remain after removing missing or invalid numeric entries. "
            "Please ensure Sales ($) and Hits contain non-negative numbers."
        )

    # Step 1: Classify Sales ($) and Hits separately
    sales_cum, sales_cls, sales_sc = assign_pareto_tier(
        clean_df["Sales ($)"],
        item_numbers=clean_df["Item Number"],
        a_cutoff=0.60,
        b_cutoff=0.80,
        c_cutoff=0.90,
    )
    hits_cum, hits_cls, hits_sc = assign_pareto_tier(
        clean_df["Hits"],
        item_numbers=clean_df["Item Number"],
        a_cutoff=0.60,
        b_cutoff=0.80,
        c_cutoff=0.90,
    )

    clean_df["Sales cumulative percentage"] = sales_cum
    clean_df["Sales Cum %"] = sales_cum  # synonym for compact UI
    clean_df["Sales Class"] = sales_cls
    clean_df["Sales Score"] = sales_sc

    clean_df["Hits cumulative percentage"] = hits_cum
    clean_df["Hits Cum %"] = hits_cum  # synonym for compact UI
    clean_df["Hits Class"] = hits_cls
    clean_df["Hits Score"] = hits_sc

    # Step 2: Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    clean_df["Weighted Score"] = (
        sales_weight * clean_df["Sales Score"] + hits_weight * clean_df["Hits Score"]
    ).round(4)

    # Step 3: Assign Final Inventory Class
    clean_df["Final Inventory Class"] = clean_df["Weighted Score"].apply(assign_final_class_from_score)
    clean_df["Multicriteria Class"] = clean_df["Final Inventory Class"]  # synonym for backwards compatibility

    # Sort descending by Sales ($) and Hits
    clean_df = clean_df.sort_values(by=["Sales ($)", "Hits"], ascending=[False, False]).reset_index(drop=True)

    # Required Summary: Count of items and total Sales ($) for each final class
    total_items = len(clean_df)
    total_sales = float(clean_df["Sales ($)"].sum())
    total_hits = float(clean_df["Hits"].sum())

    class_breakdown: Dict[str, MulticriteriaClassSummary] = {}
    for cls_name in ["A", "B", "C", "D"]:
        subset = clean_df[clean_df["Final Inventory Class"] == cls_name]
        cnt = len(subset)
        s_sum = float(subset["Sales ($)"].sum())
        h_sum = float(subset["Hits"].sum())

        class_breakdown[cls_name] = MulticriteriaClassSummary(
            class_name=cls_name,
            item_count=cnt,
            pct_items=(cnt / total_items * 100.0) if total_items > 0 else 0.0,
            total_sales=s_sum,
            pct_sales=(s_sum / total_sales * 100.0) if total_sales > 0 else 0.0,
            total_hits=h_sum,
            pct_hits=(h_sum / total_hits * 100.0) if total_hits > 0 else 0.0,
            strategy=CLASS_STRATEGIES[cls_name],
        )

    # Informational 2D cross-tabulation of Sales Class vs Hits Class
    cross_matrix = pd.crosstab(
        clean_df["Sales Class"],
        clean_df["Hits Class"],
        margins=True,
        margins_name="Total",
    ).reindex(index=["A", "B", "C", "D", "Total"], columns=["A", "B", "C", "D", "Total"], fill_value=0)

    summary = MulticriteriaClassificationSummary(
        total_items=total_items,
        total_sales=total_sales,
        total_hits=total_hits,
        class_breakdown=class_breakdown,
        cross_matrix=cross_matrix,
        warnings=warnings,
        raw_row_count=raw_row_count,
        clean_row_count=len(clean_df),
    )

    return clean_df, summary


def process_inventory_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, InventoryDatasetSummary]:
    """
    Validates, cleans, and computes inventory valuation metrics for an uploaded DataFrame
    following the valuation schema (Product, Category, Units, Unit Cost).
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
        dropped_count = int(empty_products.sum())
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
        invalid_count = int(invalid_mask.sum())
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


def load_inventory_csv_from_bytes(file_bytes_or_buffer: Any) -> Tuple[pd.DataFrame, Any]:
    """
    Helper to read bytes / uploaded file buffer directly from Streamlit's file_uploader.
    Automatically detects whether the file is a classification dataset or valuation dataset.
    """
    try:
        raw_df = pd.read_csv(file_bytes_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")

    schema = detect_dataset_schema(raw_df)
    if schema == "classification":
        return classify_inventory_multicriteria(raw_df)
    elif schema == "valuation":
        return process_inventory_dataset(raw_df)
    else:
        norm = normalize_column_names(raw_df)
        _, missing_val = validate_inventory_dataset(norm)
        _, missing_cls = validate_classification_dataset(norm)
        raise ValueError(
            f"Uploaded CSV does not match either supported inventory schema.\n"
            f"- For Multicriteria Classification, missing: {', '.join(missing_cls)}\n"
            f"- For Inventory Valuation, missing: {', '.join(missing_val)}"
        )
