"""
Unit Tests for CSV Inventory Dataset Upload, Validation, and Descriptive Analysis
=================================================================================
Validates:
1. Column schema validation for:
   - Product
   - Category
   - Units
   - Unit Cost
2. Error messaging when required columns are missing.
3. Proper handling of missing values, string currency symbols ($), commas, and invalid numerics.
4. Calculation of Inventory Value = Units * Unit Cost.
5. Descriptive analysis metrics:
   - Number of products
   - Total units
   - Total inventory value
"""

import os
import sys
import io
import pytest
import pandas as pd

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inventory_dataset import (
    validate_inventory_dataset,
    process_inventory_dataset,
    normalize_column_names,
    load_inventory_csv_from_bytes,
    REQUIRED_COLUMNS,
)


class TestInventoryDatasetValidation:
    @pytest.fixture
    def valid_raw_df(self):
        return pd.DataFrame({
            "Product": ["SKU-001", "SKU-002", "SKU-003", "SKU-004"],
            "Category": ["Electronics", "Electronics", "Hardware", "Office"],
            "Units": [100, 250, 50, 500],
            "Unit Cost": [45.00, 12.50, 8.00, 1.20],
        })

    def test_valid_dataset_passes_validation(self, valid_raw_df):
        is_valid, missing = validate_inventory_dataset(valid_raw_df)
        assert is_valid is True
        assert len(missing) == 0

    def test_valid_dataset_calculation_and_summary(self, valid_raw_df):
        clean_df, summary = process_inventory_dataset(valid_raw_df)

        # Check Inventory Value = Units * Unit Cost
        expected_values = [100 * 45.00, 250 * 12.50, 50 * 8.00, 500 * 1.20]
        assert list(clean_df["Inventory Value"]) == expected_values

        # Check Descriptive Analysis
        assert summary.num_products == 4
        assert summary.total_units == 100 + 250 + 50 + 500  # 900
        assert summary.total_inventory_value == (4500.0 + 3125.0 + 400.0 + 600.0)  # 8625.0
        assert summary.num_categories == 3
        assert len(summary.warnings) == 0

    def test_missing_single_required_column(self):
        # Missing 'Unit Cost'
        df = pd.DataFrame({
            "Product": ["A", "B"],
            "Category": ["Cat1", "Cat2"],
            "Units": [10, 20],
        })
        is_valid, missing = validate_inventory_dataset(df)
        assert is_valid is False
        assert "Unit Cost" in missing

        with pytest.raises(ValueError, match="missing required column.*Unit Cost"):
            process_inventory_dataset(df)

    def test_missing_multiple_required_columns(self):
        df = pd.DataFrame({
            "SKU": ["A", "B"],
            "Stock": [10, 20],
        })
        is_valid, missing = validate_inventory_dataset(df)
        assert is_valid is False
        assert set(missing) == set(REQUIRED_COLUMNS)

        with pytest.raises(ValueError, match="missing required column"):
            process_inventory_dataset(df)

    def test_case_insensitive_and_whitespace_column_names(self):
        df = pd.DataFrame({
            "  product  ": ["Prod-A", "Prod-B"],
            "CATEGORY": ["Tools", "Tools"],
            "units": [10, 20],
            "unit cost": ["$15.00", "$25.00"],
        })
        clean_df, summary = process_inventory_dataset(df)
        assert summary.num_products == 2
        assert summary.total_units == 30
        assert summary.total_inventory_value == (10 * 15.0 + 20 * 25.0)  # 650.0

    def test_currency_symbols_and_comma_formatting(self):
        df = pd.DataFrame({
            "Product": ["Widget A", "Widget B"],
            "Category": ["Industrial", "Industrial"],
            "Units": ["1,500", "2,000"],
            "Unit Cost": ["$1,250.50", " $45.00 "],
        })
        clean_df, summary = process_inventory_dataset(df)
        assert clean_df["Units"].iloc[0] == 1500.0
        assert clean_df["Unit Cost"].iloc[0] == 1250.50
        assert summary.total_units == 3500.0
        assert pytest.approx(summary.total_inventory_value, rel=1e-2) == (1500 * 1250.50 + 2000 * 45.00)

    def test_missing_values_and_invalid_numeric_data_handling(self):
        df = pd.DataFrame({
            "Product": ["Good-1", "Good-2", "", "Bad-Units", "Bad-Cost", "Negative-Units"],
            "Category": ["General", "General", "General", "General", "General", "General"],
            "Units": [100, 200, 50, "INVALID_NUM", 40, -10],
            "Unit Cost": [10.0, 20.0, 5.0, 15.0, "NOT_A_PRICE", 5.0],
        })
        clean_df, summary = process_inventory_dataset(df)

        # Only Good-1 (100 * 10 = 1000) and Good-2 (200 * 20 = 4000) should remain valid
        assert len(clean_df) == 2
        assert list(clean_df["Product"]) == ["Good-1", "Good-2"]
        assert summary.num_products == 2
        assert summary.total_units == 300.0
        assert summary.total_inventory_value == 5000.0
        assert len(summary.warnings) >= 1

    def test_empty_dataset_raises_error(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="uploaded CSV file is empty"):
            process_inventory_dataset(empty_df)

    def test_load_from_csv_bytes_buffer(self):
        csv_text = "Product,Category,Units,Unit Cost\nAlpha,Parts,50,12.00\nBeta,Parts,100,8.50\n"
        buf = io.BytesIO(csv_text.encode("utf-8"))
        clean_df, summary = load_inventory_csv_from_bytes(buf)

        assert summary.num_products == 2
        assert summary.total_units == 150.0
        assert summary.total_inventory_value == (50 * 12.0 + 100 * 8.5)
