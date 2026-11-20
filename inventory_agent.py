"""
Supply Chain Analytics Inventory Decision Agent
===============================================
Orchestrates the 7-step architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock and Reorder Point
4. Fixed-Period Review
5. Newsvendor
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import inventory_tools as tools


# ---------------------------------------------------------------------------
# Architecture Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class AgentResponse:
    question: str
    identified_model: str
    model_display_name: str
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool: str
    raw_result: Any
    verification: Optional[tools.VerificationResult]
    explanation: str
    pipeline_trace: List[PipelineStep] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Step 1 & 2: Identify Model
# ---------------------------------------------------------------------------

class ModelIdentifier:
    """Classifies natural-language inventory questions into supported inventory models."""

    KEYWORDS = {
        "newsvendor": [
            "newsvendor", "newsboy", "single period", "salvage", "salvaged", "perishable",
            "spoilage", "seasonal", "underage", "overage", "critical ratio",
            "critical fractile", "lost sale penalty", "holiday", "baskets"
        ],
        "quantity_discount": [
            "quantity discount", "discount tier", "price break", "price breaks",
            "quantity break", "quantity breaks", "volume discount",
            "bulk discount", "price schedule", "bracket", "brackets",
            "tier 1", "tiers", "price tier", "price tiers", "discount tiers",
            "supplier charges", "charges $", "orders below", "orders of",
            "per unit for orders", "units or more", "best order quantity"
        ],
        "fixed_period_review": [
            "fixed period", "periodic review", "review period", "order-up-to",
            "order up to", "target stock", "protection interval", "review every",
            "review inventory every", "p, s", "(p,s)", "base stock", "inventory position",
            "on order", "on-order", "on hand", "on-hand", "backorders", "backordered"
        ],
        "safety_stock_rop": [
            "safety stock", "reorder point", "rop", "continuous review", "r, q",
            "(r,q)", "service level", "cycle service level", "lead time demand", "stockout risk",
            "stock out", "z-score", "buffer stock", "lead time standard deviation", "lead time std dev"
        ],
        "eoq": [
            "eoq", "economic order quantity", "annual demand", "order cost",
            "setup cost", "holding cost", "carrying cost", "batch size",
            "lot size", "orders per year", "optimal order"
        ],
    }

    MODEL_DISPLAY_NAMES = {
        "eoq": "Economic Order Quantity (EOQ)",
        "quantity_discount": "Quantity Discount Model",
        "safety_stock_rop": "Safety Stock & Reorder Point (Continuous Review)",
        "fixed_period_review": "Fixed-Period Review (Periodic Review Order-Up-To)",
        "newsvendor": "Newsvendor Model (Single-Period Stochastic)",
    }

    @classmethod
    def identify(cls, question: str) -> Tuple[str, float, List[str]]:
        """
        Returns (model_key, confidence, matched_keywords)
        """
        q_lower = question.lower()
        scores: Dict[str, Tuple[int, List[str]]] = {}

        # Check explicit mentions first
        for model_key, keywords in cls.KEYWORDS.items():
            matched = [kw for kw in keywords if kw in q_lower]
            scores[model_key] = (len(matched), matched)

        # Check if multiple discount tiers can be extracted from the question
        detected_tiers = InputExtractor.extract_discount_tiers(question)
        if len(detected_tiers) >= 2:
            return "quantity_discount", 0.95, ["multiple_price_tiers_detected"]

        # Explicit pattern regex checks
        if re.search(r'review(?:\s+\w+){0,3}\s+every\s+\d+', q_lower) or re.search(r'on[\s-]hand.*on[\s-]order', q_lower):
            scores["fixed_period_review"] = (scores["fixed_period_review"][0] + 3, scores["fixed_period_review"][1] + ["review_every_pattern"])

        if scores["fixed_period_review"][0] > 0 and (
            "review" in q_lower or "order-up-to" in q_lower or "periodic" in q_lower or "order up to" in q_lower or "on order" in q_lower
        ):
            return "fixed_period_review", 0.95, scores["fixed_period_review"][1]

        # Prioritize more specific models over general EOQ
        if scores["quantity_discount"][0] > 0 and (
            "tier" in q_lower or "discount" in q_lower or "price break" in q_lower or "bracket" in q_lower
        ):
            return "quantity_discount", 0.95, scores["quantity_discount"][1]

        if scores["newsvendor"][0] > 0 and (
            "salvage" in q_lower or "salvaged" in q_lower or "newsvendor" in q_lower or "perishable" in q_lower or "single period" in q_lower
        ):
            return "newsvendor", 0.95, scores["newsvendor"][1]

        if scores["safety_stock_rop"][0] > 0 and (
            "safety stock" in q_lower or "reorder point" in q_lower or "rop" in q_lower or "service level" in q_lower
        ):
            return "safety_stock_rop", 0.90, scores["safety_stock_rop"][1]

        # Rank all scores
        best_model = max(scores.keys(), key=lambda k: scores[k][0])
        best_count, best_kws = scores[best_model]

        if best_count > 0:
            confidence = min(0.9, 0.4 + 0.2 * best_count)
            return best_model, confidence, best_kws

        # Default fallback to EOQ if general inventory question
        return "eoq", 0.30, ["default_fallback"]


# ---------------------------------------------------------------------------
# Step 3: Extract Inputs
# ---------------------------------------------------------------------------

class InputExtractor:
    """
    Advanced semantic entity extractor for supply chain inventory problems.
    Recognizes common terminology, synonyms, and variations.
    Examines the entire question across multiple parsing passes before declaring information missing.
    """

    @classmethod
    def extract_numbers_after(cls, text: str, patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val_str = match.group(1).replace(",", "").replace("$", "").replace("%", "")
                try:
                    return float(val_str)
                except ValueError:
                    continue
        return None

    @classmethod
    def extract_percentage(cls, text: str, patterns: List[str]) -> Optional[float]:
        val = cls.extract_numbers_after(text, patterns)
        if val is not None:
            return val / 100.0 if val > 1.0 else val
        return None

    @classmethod
    def find_all_numbers(cls, text: str) -> List[float]:
        """Finds all raw numeric values in a string (handling commas)."""
        raw = re.findall(r'\b[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\b', text)
        res = []
        for r in raw:
            try:
                res.append(float(r.replace(",", "")))
            except ValueError:
                pass
        return res

    @classmethod
    def parse_time_period(cls, period_str: str) -> str:
        """Normalizes time period string to 'year', 'month', 'week', 'day', 'quarter'."""
        p = str(period_str).lower().strip()
        if any(k in p for k in ["year", "annum", "annual", "yearly"]):
            return "year"
        elif any(k in p for k in ["month", "monthly", "mo", "mth"]):
            return "month"
        elif any(k in p for k in ["week", "weekly", "wk"]):
            return "week"
        elif any(k in p for k in ["quarter", "quarterly", "qtr"]):
            return "quarter"
        elif any(k in p for k in ["day", "daily"]):
            return "day"
        return "year"

    @classmethod
    def get_period_annual_multiplier(cls, period: str, working_days: int = 250) -> float:
        """Returns the conversion factor from period units to annual units."""
        if period == "month":
            return 12.0
        elif period == "week":
            return 52.0
        elif period == "quarter":
            return 4.0
        elif period == "day":
            return float(working_days)
        return 1.0

    @classmethod
    def extract_demand_with_period(
        cls, text: str, working_days: int = 250
    ) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
        """
        Extracts demand and recognizes sub-annual time units (day, week, month, quarter, year).
        Converts sub-annual demand to annual demand for consistent operations research calculations.

        Returns:
            (annual_demand, stated_period, raw_stated_demand, conversion_note)
        """
        # Patterns with explicit sub-annual or annual period
        patterns_with_period = [
            # 1. demand ... [number] units per/a/each [period]
            r'(?:demand|usage|sales|requirement|sells?|consumes?|requires?)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per|/|each|a)\s*(year|annum|month|week|day|quarter)',
            # 2. [number] units per/a/each [period] (requires 'units')
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units)\s*(?:per|/|each|a)\s*(year|annum|month|week|day|quarter)',
            # 3. (monthly|weekly|daily|annual|yearly|quarterly) demand ... [number]
            r'(annual|yearly|monthly|weekly|daily|quarterly)\s*(?:demand|usage|sales|requirement)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
            # 4. demand ... [number] units (annually|yearly|monthly|weekly|daily|quarterly)
            r'(?:demand|usage|sales|requirement)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(annually|yearly|monthly|weekly|daily|quarterly)',
        ]

        for pat in patterns_with_period:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                g1, g2 = match.group(1), match.group(2)
                try:
                    raw_val = float(g1.replace(",", ""))
                    p_str = g2
                except ValueError:
                    raw_val = float(g2.replace(",", ""))
                    p_str = g1

                period = cls.parse_time_period(p_str)
                mult = cls.get_period_annual_multiplier(period, working_days=working_days)
                annual_val = raw_val * mult

                note = None
                if period != "year":
                    if period == "day":
                        note = f"Stated daily demand of {raw_val:,.0f} units/day converted to annual demand of {annual_val:,.0f} units/year ({raw_val:,.0f} × {working_days} working days/year)."
                    elif period == "week":
                        note = f"Stated weekly demand of {raw_val:,.0f} units/week converted to annual demand of {annual_val:,.0f} units/year ({raw_val:,.0f} × 52 weeks/year)."
                    elif period == "month":
                        note = f"Stated monthly demand of {raw_val:,.0f} units/month converted to annual demand of {annual_val:,.0f} units/year ({raw_val:,.0f} × 12 months/year)."
                    elif period == "quarter":
                        note = f"Stated quarterly demand of {raw_val:,.0f} units/quarter converted to annual demand of {annual_val:,.0f} units/year ({raw_val:,.0f} × 4 quarters/year)."

                return annual_val, period, raw_val, note

        # Fallback to plain annual demand patterns
        plain_patterns = [
            r'(?:annual\s+demand|yearly\s+demand|annual\s+sales|yearly\s+sales|annual\s+usage|annual\s+requirement|customer\s+demand|demand\s+of|demand\s+is)\s*(?:of|is|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+per\s+year|units/year|annual\s+units|units\s+annually)',
            r'(?:sells?|selling|consumes?|requires?)\s*(?:about|around)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+year|annually|a\s+year|each\s+year)',
            r'\b[dD]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
        ]
        val = cls.extract_numbers_after(text, plain_patterns)
        if val is not None:
            return val, "year", val, None

        # Clause-level fallback
        clauses = re.split(r'[,;.\n]|\band\b', text, flags=re.IGNORECASE)
        for clause in clauses:
            c_low = clause.lower()
            if any(w in c_low for w in ["demand", "sells", "usage", "annual", "consumes"]):
                nums = cls.find_all_numbers(clause)
                if nums:
                    raw_val = nums[0]
                    period = "year"
                    if any(m in c_low for m in ["month", "monthly", "mo"]):
                        period = "month"
                    elif any(w in c_low for w in ["week", "weekly", "wk"]):
                        period = "week"
                    elif any(d in c_low for d in ["day", "daily"]):
                        period = "day"
                    elif any(q in c_low for q in ["quarter", "quarterly"]):
                        period = "quarter"

                    mult = cls.get_period_annual_multiplier(period, working_days=working_days)
                    annual_val = raw_val * mult
                    note = f"Converted {raw_val:,.0f} units/{period} to annual demand of {annual_val:,.0f} units/year." if period != "year" else None
                    return annual_val, period, raw_val, note

        return None, None, None, None

    @classmethod
    def extract_holding_cost_with_period(
        cls, text: str, working_days: int = 250
    ) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
        """
        Extracts fixed holding cost and recognizes sub-annual time units (day, week, month, year).
        Converts sub-annual holding cost to annual holding cost ($/unit/year).

        Returns:
            (annual_holding_cost, stated_period, raw_holding_cost, conversion_note)
        """
        patterns_with_period = [
            r'\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per\s+unit|/\s*unit)\s*(?:per|/|each|a)\s*(year|annum|month|week|day)',
            r'(?:fixed\s+holding\s+cost|holding\s+cost|carrying\s+cost|storage\s+cost)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per|/|each|a)\s*(year|annum|month|week|day)',
        ]
        for pat in patterns_with_period:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                raw_val = float(match.group(1).replace(",", ""))
                period = cls.parse_time_period(match.group(2))
                mult = cls.get_period_annual_multiplier(period, working_days=working_days)
                annual_val = raw_val * mult
                note = f"Stated holding cost of ${raw_val:.2f}/unit/{period} converted to annual holding cost of ${annual_val:.2f}/unit/year." if period != "year" else None
                return annual_val, period, raw_val, note

        std_pats = [
            r'(?:fixed\s+holding\s+cost|holding\s+cost|carrying\s+cost|storage\s+cost|holding\s+expense|carrying\s+expense|inventory\s+carrying\s+cost)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)',
            r'\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|per\s+unit\s+annually|/\s*unit\s*/\s*year|/\s*unit\s+annually|per\s+unit\s+holding|per\s+unit\s+carrying)',
            r'holding_cost[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            r'\b[hH]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
        ]
        val = cls.extract_numbers_after(text, std_pats)
        if val is not None:
            return val, "year", val, None

        clauses = re.split(r'[,;.\n]|\band\b', text, flags=re.IGNORECASE)
        for clause in clauses:
            c_low = clause.lower()
            if any(w in c_low for w in ["holding", "carrying", "storage"]) and "%" not in clause:
                nums = cls.find_all_numbers(clause)
                if nums:
                    raw_val = nums[0]
                    period = "year"
                    if any(m in c_low for m in ["month", "monthly", "mo"]):
                        period = "month"
                    elif any(w in c_low for w in ["week", "weekly", "wk"]):
                        period = "week"
                    elif any(d in c_low for d in ["day", "daily"]):
                        period = "day"
                    mult = cls.get_period_annual_multiplier(period, working_days=working_days)
                    annual_val = raw_val * mult
                    note = f"Converted ${raw_val:.2f}/unit/{period} to annual holding cost of ${annual_val:.2f}/unit/year." if period != "year" else None
                    return annual_val, period, raw_val, note

        return None, None, None, None

    @classmethod
    def extract_ordering_cost(cls, text: str) -> Optional[float]:
        """
        Recognizes all common ordering cost synonyms:
        - ordering cost
        - order cost
        - setup cost
        - setup expense
        - cost per order
        - cost of placing an order
        - ordering/setup cost
        - order/setup cost
        - cost to place an order
        - cost to order
        - each order costs
        - placing an order costs
        - order placement cost
        - order fee
        """
        # Primary regex patterns (prefix and suffix)
        patterns = [
            # ordering/setup cost, order/setup cost, ordering / setup cost
            r'(?:ordering\s*/\s*setup\s+cost|order\s*/\s*setup\s+cost|ordering\s*&\s*setup\s+cost)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # cost of placing an order, cost to place an order
            r'(?:cost\s+of\s+placing\s+(?:an?\s+)?order|cost\s+to\s+place\s+(?:an?\s+)?order|cost\s+of\s+(?:an?\s+)?order\s+placement)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # setup expense, setup cost
            r'(?:setup\s+expense|setup\s+cost|order\s+setup\s+cost|order\s+setup\s+expense)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # cost per order, order cost, ordering cost, order fee
            r'(?:cost\s+per\s+order|ordering\s+cost|order\s+cost|cost\s+to\s+order|order\s+placement\s+cost|order\s+fee)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # each order costs $X, placing an order costs $X
            r'(?:each\s+order\s+costs?|placing\s+an?\s+order\s+costs?|to\s+place\s+an?\s+order\s+costs?)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Suffix: $X per order, $X cost per order, $X setup cost/expense
            r'\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per\s+order|/\s*order|cost\s+per\s+order|setup\s+cost|setup\s+expense|ordering\s+cost)',
            # Explicit variable s: or S =
            r'\b[sS]\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)',
        ]
        val = cls.extract_numbers_after(text, patterns)
        if val is not None:
            return val

        # Secondary clause-level fallback: scan clauses for ordering cost synonyms
        clauses = re.split(r'[,;.\n]|\band\b', text, flags=re.IGNORECASE)
        synonyms = [
            "ordering/setup cost", "order/setup cost", "cost of placing an order",
            "cost to place an order", "cost per order", "setup expense",
            "ordering cost", "order cost", "setup cost", "cost to order",
            "order fee", "placing an order"
        ]
        for clause in clauses:
            c_low = clause.lower()
            if any(syn in c_low for syn in synonyms):
                nums = cls.find_all_numbers(clause)
                if nums:
                    return nums[0]

        return None

    @classmethod
    def extract_unit_purchase_cost(cls, text: str) -> Optional[float]:
        """
        Extracts unit purchase price / cost (C) while strictly ensuring that
        holding costs ($/unit/year, $/unit annually) or ordering costs ($/order)
        are NEVER falsely extracted as unit purchase price.
        """
        # Priority 1: Highly specific purchase price patterns
        specific_patterns = [
            r'(?:unit\s+purchase\s+(?:price|cost)|unit\s+purchase\s+cost|purchase\s+price|purchase\s+cost|wholesale\s+price)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'unit_cost[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            r'\b[cC]\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'(?:each\s+(?:unit|item)\s+costs?|items?\s+costs?|product\s+costs?)\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*(?:%|per\s+order|per\s+year|annually|/\s*year))',
            r'(?:buy|purchase)\s+(?:each\s+unit|each\s+item|units?|items?)\s*(?:for|at)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
        ]
        for pat in specific_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Priority 2: Clause-level fallback. Split text into logical clauses.
        clauses = re.split(r'[,;.\n]|\band\b', text, flags=re.IGNORECASE)
        for clause in clauses:
            c_low = clause.lower()
            # Strictly ignore clauses containing holding/carrying/storage/annual parameters
            if any(w in c_low for w in ["holding", "carrying", "storage", "per unit per year", "/ unit / year", "/unit/year", "per unit annually", "/ unit annually", "/unit annually"]):
                continue
            # Strictly ignore clauses containing ordering/setup cost terms
            if any(w in c_low for w in ["order cost", "ordering cost", "setup cost", "setup expense", "per order", "/ order", "/order", "placing an order"]):
                continue
            # Strictly ignore clauses describing demand or volume
            if any(w in c_low for w in ["demand", "sells", "sales", "requires", "consumes"]):
                continue

            # Within a clean clause that does NOT talk about holding or ordering:
            clause_patterns = [
                r'(?:unit\s+cost|cost\s+per\s+unit|item\s+cost)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*(?:%|per\s+year|annually|/\s*year))',
                r'(?:costs?|buy\s+at|purchased?\s+at)\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit|/\s*unit|each)(?!\s*(?:per\s+year|annually|/\s*year))',
                r'\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit|/\s*unit|each)(?!\s*(?:per\s+year|annually|/\s*year|per\s+order|/\s*order))',
            ]
            for pat in clause_patterns:
                match = re.search(pat, clause, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1).replace(",", ""))
                    except ValueError:
                        pass

        return None

    @classmethod
    def extract_discount_tiers(cls, text: str) -> List[Tuple[float, Optional[float], float]]:
        """
        Parses discount tiers across various natural language forms:
        - Price first with below/to/more:
          "$12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for orders of 1,000 units or more"
        - Quantity first:
          "1 to 999 units at $5.00, 1000 to 1999 units at $4.80, and 2000+ units at $4.75"
        - Colon format:
          "1-999: $5.00, 1000-1999: $4.80, 2000+: $4.75"
        - At format:
          "0-999 @ $5, 1000-1999 @ $4.80, 2000+ @ $4.60"
        - Tier labels:
          "Tier 1: 0-499 units at $10.00; Tier 2: 500-999 units at $9.00; Tier 3: 1000+ units at $8.00"
        - Bracket format:
          "[0, 499, 12.0], [500, 999, 11.50], [1000, None, 11.0]"
        """
        tiers: List[Tuple[float, Optional[float], float]] = []

        # Format 1: Bracket format [min, max, price]
        bracket_matches = re.findall(
            r'\[\s*(\d+[\d,]*)\s*,\s*(\d+[\d,]*|None|inf|\+)?\s*,\s*\$?([0-9.]+)\s*\]', text, re.IGNORECASE
        )
        if bracket_matches:
            for min_q, max_q, pr in bracket_matches:
                min_val = float(min_q.replace(",", ""))
                max_val = None if (not max_q or max_q.lower() in ["none", "inf", "+"] or max_q == "") else float(max_q.replace(",", ""))
                tiers.append((min_val, max_val, float(pr)))
            return sorted(tiers, key=lambda t: t[0])

        # Clean text: remove commas inside numbers (e.g. 5,000 -> 5000, 1,000 -> 1000)
        clean_text = re.sub(r'(\d),(\d)', r'\1\2', text)

        # Split into candidate clauses by semicolon, newline, "and" (not followed by over/above/more), or punctuation separating items
        clauses = re.split(r'[;\n]|\band\b(?!\s+(?:above|over|more))|,(?!\d)|\.(?!\d)', clean_text, flags=re.IGNORECASE)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Look for a price in the clause
            # Must have $ sign, or "dollars", or preceded by "at / @ / charges / costs / price", or followed by "per unit / each"
            pr_match = re.search(
                r'\$([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)|\b(?:at|@|costs?|charges?|price\s*[:=])\s*\$?([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|/\s*unit|each)',
                clause, re.IGNORECASE
            )
            if not pr_match:
                continue

            pr_str = pr_match.group(1) or pr_match.group(2) or pr_match.group(3) or pr_match.group(4)
            try:
                price = float(pr_str)
            except (ValueError, TypeError):
                continue

            # Remove the matched price substring to isolate the quantity range
            pr_span = pr_match.span()
            clause_rem = clause[:pr_span[0]] + " " + clause[pr_span[1]:]

            # Check quantity patterns in the remaining clause:
            # A. Below / under / less than / up to / < max_q
            m_below = re.search(r'(?:below|under|less\s+than|<|up\s+to)\s*(\d+)', clause_rem, re.IGNORECASE)
            if m_below:
                max_q = float(m_below.group(1))
                upper = max_q - 1.0 if max_q.is_integer() else max_q
                tiers.append((0.0, upper, price))
                continue

            # B. Range: min to/hyphen max
            m_range = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)', clause_rem, re.IGNORECASE)
            if m_range:
                min_q = float(m_range.group(1))
                max_q = float(m_range.group(2))
                if max_q >= min_q:
                    tiers.append((min_q, max_q, price))
                    continue

            # C. Plus / or more / and above / and over / over / above:
            m_plus = re.search(r'(\d+)\s*(?:units)?\s*(?:\+|and\s+(?:above|over|more)|or\s+more|\babove\b|\bover\b)', clause_rem, re.IGNORECASE)
            if m_plus:
                min_q = float(m_plus.group(1))
                tiers.append((min_q, None, price))
                continue

        # Deduplicate and sort by min_quantity
        unique_tiers = []
        seen = set()
        for min_q, max_q, pr in tiers:
            key = (min_q, max_q, pr)
            if key not in seen:
                seen.add(key)
                unique_tiers.append((min_q, max_q, pr))

        return sorted(unique_tiers, key=lambda t: t[0]) if unique_tiers else []

    @classmethod
    def extract_inputs(cls, model_key: str, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts parameters required for the designated inventory model.
        Returns (extracted_dict, missing_fields_list)
        """
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # =====================================================================
        # MODEL 1: EOQ
        # =====================================================================
        if model_key == "eoq":
            # Working days first (needed if daily demand conversion is required)
            w = cls.extract_numbers_after(question, [
                r'([0-9]+)\s*(?:working\s+days|operating\s+days|days\s+per\s+year|days\s+a\s+year)',
                r'working_days[:=]\s*([0-9]+)',
            ])
            working_days_val = int(w) if w else 250

            # 1. Demand D (with period recognition: day, week, month, year)
            annual_d, d_period, raw_d, d_note = cls.extract_demand_with_period(question, working_days=working_days_val)

            # 2. Ordering Cost S (Recognizing all requested synonyms)
            s = cls.extract_ordering_cost(question)

            # 3. Holding Cost H or Holding Rate i (%)
            # Check percentage holding cost first
            rate = cls.extract_percentage(question, [
                r'(?:holding\s+rate|carrying\s+rate|holding\s+cost\s+rate|carrying\s+cost\s+rate|annual\s+holding\s+rate|carrying\s+percentage)\s*(?:is|of|[:=]|about)?\s*([0-9.]+)\s*%?',
                r'(?:percentage\s+holding\s+cost|holding\s+cost|carrying\s+cost)\s*(?:is|of|[:=]|about)?\s*([0-9.]+)\s*%',
                r'([0-9.]+)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?(?:cost|price|purchase\s+price)|annual\s*holding|carrying\s*cost|holding\s*cost|holding\s*rate)',
                r'holding_rate[:=]\s*([0-9.]+)',
            ])

            annual_h, h_period, raw_h, h_note = None, None, None, None
            if rate is None:
                annual_h, h_period, raw_h, h_note = cls.extract_holding_cost_with_period(question, working_days=working_days_val)

            # 4. Unit purchase price C (strictly separate from holding cost $/unit/year)
            c = cls.extract_unit_purchase_cost(question)

            if annual_d is not None:
                inputs["demand"] = annual_d
                if d_period and d_period != "year":
                    inputs["stated_demand"] = raw_d
                    inputs["demand_period"] = d_period
                    inputs["demand_conversion_note"] = d_note
            else:
                missing.append("demand (Annual Demand D)")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (Ordering/Setup Cost S: ordering cost, setup cost, cost per order, setup expense, cost of placing an order)")

            if annual_h is not None:
                inputs["holding_cost"] = annual_h
                if h_period and h_period != "year":
                    inputs["stated_holding_cost"] = raw_h
                    inputs["holding_period"] = h_period
                    inputs["holding_conversion_note"] = h_note
            elif rate is not None and c is not None:
                inputs["holding_rate"] = rate
                inputs["unit_cost"] = c
            elif rate is not None and c is None:
                inputs["holding_rate"] = rate
                missing.append("unit_cost (Unit Purchase Price C needed to apply holding rate)")
            else:
                missing.append("holding_cost (Holding Cost H or rate i% with unit purchase price C)")

            if c is not None and "unit_cost" not in inputs:
                inputs["unit_cost"] = c
            inputs["working_days"] = working_days_val

        # =====================================================================
        # MODEL 2: QUANTITY DISCOUNT
        # =====================================================================
        elif model_key == "quantity_discount":
            w = cls.extract_numbers_after(question, [
                r'([0-9]+)\s*(?:working\s+days|operating\s+days|days\s+per\s+year|days\s+a\s+year)',
                r'working_days[:=]\s*([0-9]+)',
            ])
            working_days_val = int(w) if w else 250

            annual_d, d_period, raw_d, d_note = cls.extract_demand_with_period(question, working_days=working_days_val)
            s = cls.extract_ordering_cost(question)

            rate = cls.extract_percentage(question, [
                r'(?:percentage\s+holding\s+cost|holding\s+rate|carrying\s+rate|holding\s+cost\s+rate|carrying\s+cost\s+rate)\s*(?:is|of|[:=]|about)?\s*([0-9.]+)\s*%?',
                r'(?:annual\s+)?(?:holding\s+cost|carrying\s+cost)\s*(?:is|of|[:=]|about)?\s*([0-9.]+)\s*%',
                r'([0-9.]+)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?(?:cost|price)|annual\s*holding|carrying\s*rate)',
                r'holding_rate[:=]\s*([0-9.]+)',
            ])
            annual_h, h_period, raw_h, h_note = None, None, None, None
            if rate is None:
                annual_h, h_period, raw_h, h_note = cls.extract_holding_cost_with_period(question, working_days=working_days_val)

            base_p = cls.extract_numbers_after(question, [
                r'(?:base\s+price|unit\s+purchase\s+price|standard\s+price|base\s+cost)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            ])

            tiers = cls.extract_discount_tiers(question)

            if annual_d is not None:
                inputs["demand"] = annual_d
                if d_period and d_period != "year":
                    inputs["stated_demand"] = raw_d
                    inputs["demand_period"] = d_period
                    inputs["demand_conversion_note"] = d_note
            else:
                missing.append("demand (Annual Demand)")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (Ordering Cost / Setup Cost)")

            if rate is not None:
                inputs["holding_rate"] = rate
            elif annual_h is not None:
                inputs["holding_cost"] = annual_h
                if h_period and h_period != "year":
                    inputs["stated_holding_cost"] = raw_h
                    inputs["holding_period"] = h_period
                    inputs["holding_conversion_note"] = h_note
            else:
                missing.append("holding_cost or holding_rate (Fixed holding cost or percentage holding cost)")

            if tiers:
                inputs["discount_tiers"] = tiers
            else:
                missing.append("discount_tiers (Quantity breaks and unit prices, e.g. 1-999: $5.00, 1000-1999: $4.80, 2000+: $4.75)")

            if base_p is not None:
                inputs["unit_purchase_price"] = base_p

        # =====================================================================
        # MODEL 3: SAFETY STOCK & REORDER POINT
        # =====================================================================
        elif model_key == "safety_stock_rop":
            w = cls.extract_numbers_after(question, [
                r'([0-9]+)\s*(?:working\s+days|operating\s+days|days\s+per\s+year|days\s+a\s+year)',
                r'working_days[:=]\s*([0-9]+)',
            ])
            working_days_val = int(w) if w else 250

            # -------------------------------------------------------------
            # 1. Lead Time L and Lead Time Std Dev sigma_L (with period recognition)
            # -------------------------------------------------------------
            lt, lt_unit, lt_std, lt_std_unit = None, None, None, None

            # Combined lead time pattern: e.g. "average lead time is 3 weeks with a standard deviation of 0.5 weeks"
            comb_lt = re.search(
                r'(?:average\s+lead[\s-]time|mean\s+lead[\s-]time|supplier\s+lead[\s-]time|replenishment\s+lead[\s-]time|lead[\s-]time)\s*(?:is|averages?|of|takes|[:=])?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?\s*(?:with\s+(?:a\s+)?)?(?:standard\s+deviation|std\s*dev)\s*(?:of|is|[:=])?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                question, re.IGNORECASE
            )
            if comb_lt:
                lt = float(comb_lt.group(1).replace(",", ""))
                lt_unit = cls.parse_time_period(comb_lt.group(2)) if comb_lt.group(2) else None
                lt_std = float(comb_lt.group(3).replace(",", ""))
                lt_std_unit = cls.parse_time_period(comb_lt.group(4)) if comb_lt.group(4) else lt_unit
            else:
                # Separate lead time std dev
                lt_std_pats = [
                    r'(?:standard\s+deviation\s+of\s+(?:the\s+)?lead[\s-]time|lead[\s-]time\s+standard\s+deviation|lead[\s-]time\s+std\s*dev|std\s*dev\s+of\s+(?:the\s+)?lead[\s-]time|lead[\s-]time\s+variability|sigma_L|sigma\s+l)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                    r'lead[\s-]time(?:\s+\w+){0,6}\s+(?:standard\s+deviation|std\s*dev)\s*(?:is|of|[:=])?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                    r'lead_time_std[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                ]
                for pat in lt_std_pats:
                    m = re.search(pat, question, re.IGNORECASE)
                    if m:
                        lt_std = float(m.group(1).replace(",", ""))
                        lt_std_unit = cls.parse_time_period(m.group(2)) if (m.lastindex and m.lastindex >= 2 and m.group(2)) else None
                        break

                # Separate lead time mean
                lt_pats = [
                    r'(?:average\s+lead[\s-]time|mean\s+lead[\s-]time|supplier\s+lead[\s-]time|replenishment\s+lead[\s-]time|lead[\s-]time)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                    r'takes\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)\s*(?:to\s+(?:arrive|deliver|replenish)|lead[\s-]time)?',
                    r'([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)\s*(?:supplier\s+)?lead[\s-]time',
                    r'lead_time[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                    r'\b[lL]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(days?|weeks?|months?)?',
                ]
                for pat in lt_pats:
                    m = re.search(pat, question, re.IGNORECASE)
                    if m:
                        lt = float(m.group(1).replace(",", ""))
                        lt_unit = cls.parse_time_period(m.group(2)) if (m.lastindex and m.lastindex >= 2 and m.group(2)) else None
                        break

            # -------------------------------------------------------------
            # 2. Demand Mean d and Demand Std Dev sigma_d
            # -------------------------------------------------------------
            d_mean, d_unit, d_std, d_std_unit = None, None, None, None

            # Combined demand pattern: e.g. "average demand of 40 units per day with a standard deviation of 8 units per day"
            comb_d = re.search(
                r'(?:average\s+(?:daily\s+)?demand|mean\s+(?:daily\s+)?demand|daily\s+demand\s+mean|expected\s+(?:daily\s+)?demand|daily\s+demand|average\s+demand|mean\s+demand|demand)\s*(?:is|averages?|of|[:=])?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?\s*(?:with\s+(?:a\s+)?)?(?:standard\s+deviation|std\s*dev)\s*(?:of|is|[:=])?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?',
                question, re.IGNORECASE
            )
            if comb_d:
                d_mean = float(comb_d.group(1).replace(",", ""))
                d_unit = cls.parse_time_period(comb_d.group(2)) if comb_d.group(2) else None
                d_std = float(comb_d.group(3).replace(",", ""))
                d_std_unit = cls.parse_time_period(comb_d.group(4)) if comb_d.group(4) else d_unit
            else:
                # Separate demand std
                d_std_pats = [
                    r'(?:standard\s+deviation\s+of\s+(?:daily\s+)?demand|demand\s+standard\s+deviation|demand\s+std\s*dev|std\s*dev\s+of\s+(?:daily\s+)?demand|daily\s+demand\s+std\s*dev|demand\s+variability|sigma_d|sigma\s+d)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?',
                    r'with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)(?:\s+of)?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?',
                    r'demand_std[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?',
                ]
                for pat in d_std_pats:
                    m = re.search(pat, question, re.IGNORECASE)
                    if m:
                        d_std = float(m.group(1).replace(",", ""))
                        d_std_unit = cls.parse_time_period(m.group(2)) if (m.lastindex and m.lastindex >= 2 and m.group(2)) else None
                        break

                # Separate demand mean
                d_pats = [
                    r'(?:average\s+(?:daily\s+)?demand|mean\s+(?:daily\s+)?demand|daily\s+demand\s+mean|expected\s+(?:daily\s+)?demand|daily\s+demand|average\s+demand|mean\s+demand)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units)?\s*(?:per|/|each|a)?\s*(day|week|month|year)?',
                    r'demand\s+(?:averages|is)\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units\s+per\s+day|units/day|per\s+day|units\s+per\s+week|units/week|per\s+week)?',
                    r'([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:units\s+per\s+day|units/day)\s*(?:demand|average)?',
                    r'demand_mean[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)',
                    r'\b[dD]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?|\.[0-9]+)',
                ]
                for pat in d_pats:
                    m = re.search(pat, question, re.IGNORECASE)
                    if m:
                        d_mean = float(m.group(1).replace(",", ""))
                        d_unit = cls.parse_time_period(m.group(2)) if (m.lastindex and m.lastindex >= 2 and m.group(2)) else None
                        break

            # Fallback for demand in clauses
            if d_mean is None or (d_std is None and "constant" not in question.lower() and "(constant)" not in question.lower()):
                for clause in re.split(r'[,;.\n]|\band\b', question, flags=re.IGNORECASE):
                    c_low = clause.lower()
                    if any(w in c_low for w in ["lead time", "lead-time", "supplier", "review", "arrive", "service level"]):
                        continue
                    if ("std" in c_low or "deviation" in c_low) and d_std is None:
                        nums = cls.find_all_numbers(clause)
                        if nums:
                            d_std = nums[0]
                    elif "demand" in c_low and d_mean is None:
                        nums = cls.find_all_numbers(clause)
                        if nums:
                            d_mean = nums[0]

            # -------------------------------------------------------------
            # 3. Service Level CSL or Z
            # -------------------------------------------------------------
            csl = cls.extract_percentage(question, [
                r'(?:cycle\s+)?service\s+level\s*(?:is|of|[:=]|about)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?',
                r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level',
                r'(?:in[\s-]stock\s+probability|target\s+service\s+level|desired\s+service\s+level)\s*(?:is|of|[:=]|about)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?',
                r'service_level[:=]\s*([0-9]+(?:\.[0-9]+)?)',
            ])
            z_val = cls.extract_numbers_after(question, [
                r'\b[zZ]\s*(?:score)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
            ])

            # -------------------------------------------------------------
            # 4. Consistent Time Unit Conversion (Lead time -> Demand period)
            # -------------------------------------------------------------
            base_unit = d_unit or "day"
            days_per_week = 5.0
            days_per_month = float(working_days_val) / 12.0

            lt_mult = 1.0
            if lt_unit == "week":
                lt_mult = days_per_week
            elif lt_unit == "month":
                lt_mult = days_per_month

            lt_std_mult = 1.0
            actual_lt_std_unit = lt_std_unit or lt_unit
            if actual_lt_std_unit == "week":
                lt_std_mult = days_per_week
            elif actual_lt_std_unit == "month":
                lt_std_mult = days_per_month

            effective_lt = (lt * lt_mult) if lt is not None else None
            effective_lt_std = (lt_std * lt_std_mult) if lt_std is not None else 0.0

            conversion_note = None
            if lt_unit and lt_unit != "day":
                conversion_note = (
                    f"Stated lead time of {lt:g} {lt_unit}s converted to {effective_lt:g} days "
                    f"({lt:g} × {days_per_week:g} working days/week)."
                )
                if lt_std is not None and lt_std > 0 and actual_lt_std_unit != "day":
                    conversion_note += (
                        f" Lead time std dev converted from {lt_std:g} {actual_lt_std_unit}s to {effective_lt_std:g} days "
                        f"({lt_std:g} × {days_per_week:g} working days/week)."
                    )

            if d_mean is not None:
                inputs["demand_mean"] = d_mean
            else:
                missing.append("demand_mean (Average demand d)")

            inputs["demand_std"] = d_std if d_std is not None else 0.0
            inputs["demand_unit"] = base_unit

            if effective_lt is not None:
                inputs["lead_time"] = effective_lt
            else:
                missing.append("lead_time (Average lead time L)")

            inputs["lead_time_std"] = effective_lt_std

            if lt_unit and lt_unit != "day":
                inputs["stated_lead_time"] = lt
                inputs["stated_lead_time_unit"] = lt_unit
                if lt_std is not None and lt_std > 0:
                    inputs["stated_lead_time_std"] = lt_std
                    inputs["stated_lead_time_std_unit"] = actual_lt_std_unit
                inputs["conversion_note"] = conversion_note

            if csl is not None:
                inputs["service_level"] = csl
            elif z_val is not None:
                inputs["z"] = z_val
            else:
                if inputs["demand_std"] > 0 or inputs["lead_time_std"] > 0:
                    inputs["service_level"] = 0.95
                else:
                    inputs["service_level"] = 1.0

        # =====================================================================
        # MODEL 4: FIXED-PERIOD REVIEW
        # =====================================================================
        elif model_key == "fixed_period_review":
            # 1. Review period P
            p = cls.extract_numbers_after(question, [
                r'(?:review\s+period|review\s+interval|review\s+cycle|periodic\s+review|period\s+between\s+orders|order\s+interval)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)?',
                r'(?:review(?:ed)?|orders?\s+placed)\s+(?:inventory\s+)?every\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)',
                r'every\s*([0-9,]+(?:\.[0-9]+)?)\s*days?',
                r'review_period[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[pP]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 2. Lead time L
            lt = cls.extract_numbers_after(question, [
                r'(?:supplier\s+lead[\s-]time|delivery\s+time|lead[\s-]time)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)?',
                r'takes\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:to\s+(?:arrive|deliver))?',
                r'lead_time[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[lL]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 3. Demand mean d
            d_mean = cls.extract_numbers_after(question, [
                r'(?:daily\s+demand|average\s+demand|mean\s+demand|demand\s+mean)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units/day|units\s+per\s+day)',
                r'demand_mean[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[dD]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 4. Demand std sigma_d
            d_std = cls.extract_numbers_after(question, [
                r'(?:standard\s+deviation|std\s*dev|sigma_d)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)(?:\s+of)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'demand_std[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 5. Service level
            csl = cls.extract_percentage(question, [
                r'service\s+level\s*(?:is|of|[:=]|about)?\s*([0-9.]+)\s*%?',
                r'([0-9.]+)\s*%\s*service\s+level',
                r'service_level[:=]\s*([0-9.]+)',
            ])

            # 6. Safety stock (if stated directly)
            ss_dir = cls.extract_numbers_after(question, [
                r'(?:safety\s+stock|buffer\s+stock)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'safety_stock[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 7. Inventory position (if stated directly)
            ip_dir = cls.extract_numbers_after(question, [
                r'(?:inventory\s+position|current\s+inventory\s+position|net\s+inventory\s+position)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'inventory_position[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 8. On-hand inventory
            oh = cls.extract_numbers_after(question, [
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:on[\s-]hand|in\s+stock|physical\s+inventory)',
                r'(?:on[\s-]hand\s+inventory|on[\s-]hand|current\s+stock|current\s+inventory|in\s+stock|stock\s+on\s+hand)\s*(?:is|of|we\s+have|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'on_hand[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 9. On-order inventory
            oo = cls.extract_numbers_after(question, [
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:on[\s-]order|pipeline|in[\s-]transit|due[\s-]in|ordered|incoming)',
                r'(?:on[\s-]order\s+inventory|on[\s-]order|pipeline\s+inventory|pipeline\s+stock|in[\s-]transit|incoming|open\s+orders|ordered)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'on_order[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 10. Backorders
            bo = cls.extract_numbers_after(question, [
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:backorders?|backordered|backlog|unfilled\s+orders)',
                r'(?:backorders?|backordered|backlog|unfilled\s+orders|shortage)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'backorders[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # Clause-level fallback for FPR
            if oh is None or oo is None or bo is None:
                for clause in re.split(r'[,;.\n]|\band\b', question, flags=re.IGNORECASE):
                    c_low = clause.lower()
                    if ("on-hand" in c_low or "on hand" in c_low or "in stock" in c_low) and oh is None:
                        nums = cls.find_all_numbers(clause)
                        if nums: oh = nums[0]
                    elif ("on-order" in c_low or "on order" in c_low or "pipeline" in c_low) and oo is None:
                        nums = cls.find_all_numbers(clause)
                        if nums: oo = nums[0]
                    elif ("backorder" in c_low or "backlog" in c_low) and bo is None:
                        nums = cls.find_all_numbers(clause)
                        if nums: bo = nums[0]

            if p is not None:
                inputs["review_period"] = p
            else:
                missing.append("review_period (Review period P in days)")

            if lt is not None:
                inputs["lead_time"] = lt
            else:
                missing.append("lead_time (Lead time L in days)")

            if d_mean is not None:
                inputs["demand_mean"] = d_mean
            else:
                missing.append("demand_mean (Average daily demand d)")

            inputs["demand_std"] = d_std if d_std is not None else 0.0
            inputs["service_level"] = csl if csl is not None else 0.95
            inputs["on_hand"] = oh if oh is not None else 0.0
            inputs["on_order"] = oo if oo is not None else 0.0
            inputs["backorders"] = bo if bo is not None else 0.0

            if ip_dir is not None:
                inputs["inventory_position"] = ip_dir
            if ss_dir is not None:
                inputs["safety_stock"] = ss_dir

        # =====================================================================
        # MODEL 5: NEWSVENDOR
        # =====================================================================
        elif model_key == "newsvendor":
            # 1. Selling price p
            p = cls.extract_numbers_after(question, [
                r'(?:selling\s+price|retail\s+price|sales\s+price|retail\s+value|price\s+per\s+unit|price\s+is)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'(?:sells?|selling|sold)\s*(?:\w+\s+){0,6}(?:for|at)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'sells?\s+(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:selling\s+price|retail\s+price|each\s+retail)',
                r'price[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[pP]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 2. Purchase cost c
            c = cls.extract_numbers_after(question, [
                r'(?:purchase\s+cost|production\s+cost|manufacturing\s+cost|acquisition\s+cost|unit\s+cost|cost\s+per\s+unit)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'costs?(?:\s+\w+){0,4}\s+(?:to\s+(?:produce|buy|make|manufacture|acquire))\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s+(?:make|produce|buy)|each|per\s+unit)',
                r'cost[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[cC]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 3. Salvage value s
            s = cls.extract_numbers_after(question, [
                r'(?:salvage\s+value|salvage\s+price|scrap\s+value|liquidation\s+value|leftover\s+value)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'(?:salvage(?:d)?|liquidat(?:ed)?|scrapped|sold\s+off|discounted)(?:\s+\w+){0,4}\s+(?:for|at|value|is|of)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'salvage_value[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[sS]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 4. Underage / shortage cost Cu
            cu = cls.extract_numbers_after(question, [
                r'(?:underage\s+cost|cost\s+of\s+underage|shortage\s+cost|cost\s+of\s+shortage)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'underage_cost[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[cC]u\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 5. Overage / excess cost Co
            co = cls.extract_numbers_after(question, [
                r'(?:overage\s+cost|cost\s+of\s+overage|excess\s+cost|cost\s+of\s+excess)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'overage_cost[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[cC]o\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 6. Shortage penalty g
            g = cls.extract_numbers_after(question, [
                r'(?:shortage\s+penalty|lost\s+sales?\s+penalty|penalty\s+cost|goodwill\s+loss)\s*(?:is|of|[:=]|about)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'shortage_penalty[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 7. Mean demand mu
            mu = cls.extract_numbers_after(question, [
                r'(?:mean\s+demand|average\s+demand|expected\s+demand|demand\s+(?:is\s+normally\s+distributed\s+with\s+)?mean\s*(?:of|is)?)\s*([0-9,]+(?:\.[0-9]+)?)',
                r'(?:mean|average)\s*(?:demand)?\s*(?:is|of|[:=])\s*([0-9,]+(?:\.[0-9]+)?)',
                r'demand_mean[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[mu\u03bc]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 8. Demand standard deviation sigma
            sigma = cls.extract_numbers_after(question, [
                r'(?:standard\s+deviation\s+of\s+demand|demand\s+standard\s+deviation|demand\s+std\s*dev|std\s*dev\s+of\s+demand)\s*(?:is|of|[:=]|about)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)(?:\s+of)?\s*([0-9,]+(?:\.[0-9]+)?)',
                r'(?:standard\s+deviation|std\s*dev|sigma)\s*(?:is|of|[:=])\s*([0-9,]+(?:\.[0-9]+)?)',
                r'demand_std[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b[\u03c3]\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)',
            ])

            # 9. Distribution
            dist = "normal"
            if "uniform" in question.lower():
                dist = "uniform"

            # Clause-level fallback for Newsvendor
            if (p is None and cu is None) or (c is None and co is None):
                for clause in re.split(r'[,;.\n]|\band\b', question, flags=re.IGNORECASE):
                    c_low = clause.lower()
                    if ("sell" in c_low or "price" in c_low) and p is None:
                        nums = cls.find_all_numbers(clause)
                        if nums: p = nums[0]
                    elif ("cost" in c_low or "produce" in c_low or "make" in c_low) and c is None:
                        nums = cls.find_all_numbers(clause)
                        if nums: c = nums[0]

            # Validate parameter sets
            if cu is not None and co is not None:
                inputs["underage_cost"] = cu
                inputs["overage_cost"] = co
                if p is not None: inputs["price"] = p
                if c is not None: inputs["cost"] = c
            else:
                if p is not None:
                    inputs["price"] = p
                else:
                    missing.append("price (Selling price p or underage cost Cu)")

                if c is not None:
                    inputs["cost"] = c
                else:
                    missing.append("cost (Unit purchase/production cost c or overage cost Co)")

            if mu is not None:
                inputs["demand_mean"] = mu
            else:
                missing.append("demand_mean (Expected demand mean)")

            if sigma is not None:
                inputs["demand_std"] = sigma
            else:
                missing.append("demand_std (Demand standard deviation)")

            inputs["salvage_value"] = s if s is not None else 0.0
            inputs["shortage_penalty"] = g if g is not None else 0.0
            inputs["distribution"] = dist

        return inputs, missing


# ---------------------------------------------------------------------------
# Step 4 & 5: Tool Selection and Calculation
# ---------------------------------------------------------------------------

class ToolSelector:
    """Selects and executes the appropriate mathematical tool from inventory_tools.py."""

    TOOL_NAMES = {
        "eoq": "tools.calculate_eoq",
        "quantity_discount": "tools.calculate_quantity_discount",
        "safety_stock_rop": "tools.calculate_safety_stock_rop",
        "fixed_period_review": "tools.calculate_fixed_period_review",
        "newsvendor": "tools.calculate_newsvendor",
    }

    @classmethod
    def execute(cls, model_key: str, inputs: Dict[str, Any]) -> Tuple[str, Any]:
        tool_name = cls.TOOL_NAMES.get(model_key, "unknown_tool")

        if model_key == "eoq":
            result = tools.calculate_eoq(
                demand=inputs["demand"],
                order_cost=inputs["order_cost"],
                holding_cost=inputs.get("holding_cost"),
                unit_cost=inputs.get("unit_cost"),
                holding_rate=inputs.get("holding_rate"),
                working_days=inputs.get("working_days", 250),
                stated_demand=inputs.get("stated_demand"),
                demand_period=inputs.get("demand_period"),
                stated_holding_cost=inputs.get("stated_holding_cost"),
                holding_period=inputs.get("holding_period"),
            )
            return tool_name, result

        elif model_key == "quantity_discount":
            result = tools.calculate_quantity_discount(
                demand=inputs["demand"],
                order_cost=inputs["order_cost"],
                discount_tiers=inputs["discount_tiers"],
                holding_cost=inputs.get("holding_cost"),
                holding_rate=inputs.get("holding_rate"),
                stated_demand=inputs.get("stated_demand"),
                demand_period=inputs.get("demand_period"),
            )
            return tool_name, result

        elif model_key == "safety_stock_rop":
            result = tools.calculate_safety_stock_rop(
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"],
                lead_time=inputs["lead_time"],
                service_level=inputs.get("service_level", 0.95),
                lead_time_std=inputs.get("lead_time_std", 0.0),
                z=inputs.get("z"),
                stated_lead_time=inputs.get("stated_lead_time"),
                stated_lead_time_unit=inputs.get("stated_lead_time_unit"),
                stated_lead_time_std=inputs.get("stated_lead_time_std"),
                stated_lead_time_std_unit=inputs.get("stated_lead_time_std_unit"),
                demand_unit=inputs.get("demand_unit", "day"),
            )
            return tool_name, result

        elif model_key == "fixed_period_review":
            result = tools.calculate_fixed_period_review(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"],
                service_level=inputs.get("service_level", 0.95),
                on_hand=inputs.get("on_hand", 0.0),
                on_order=inputs.get("on_order", 0.0),
                backorders=inputs.get("backorders", 0.0),
                inventory_position=inputs.get("inventory_position"),
                safety_stock=inputs.get("safety_stock"),
            )
            return tool_name, result

        elif model_key == "newsvendor":
            result = tools.calculate_newsvendor(
                price=inputs.get("price"),
                cost=inputs.get("cost"),
                salvage_value=inputs.get("salvage_value", 0.0),
                shortage_penalty=inputs.get("shortage_penalty", 0.0),
                underage_cost=inputs.get("underage_cost"),
                overage_cost=inputs.get("overage_cost"),
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"],
                distribution=inputs.get("distribution", "normal"),
            )
            return tool_name, result

        else:
            raise ValueError(f"Unrecognized model key: {model_key}")


# ---------------------------------------------------------------------------
# Step 6 & 7: Verification and Explanation Generator
# ---------------------------------------------------------------------------

class Explainer:
    """Generates structured, clear, and actionable managerial explanations."""

    @classmethod
    def explain(cls, model_key: str, result: Any, inputs: Dict[str, Any]) -> str:
        lines: List[str] = []

        if model_key == "eoq":
            res: tools.EOQResult = result
            lines.append("## Executive Summary: Economic Order Quantity (EOQ) Recommendation")
            lines.append(f"- **Optimal Order Quantity (Q*)**: **{res.recommended_order_quantity:,} units** ({res.optimal_order_quantity:.2f} exact)")
            lines.append(f"- **Order Frequency**: Place **{res.orders_per_year:.2f} orders per year** (every **{res.cycle_time_days:.1f} working days**)")
            lines.append(f"- **Total Annual Inventory Cost**: **${res.annual_inventory_cost:,.2f}**")
            if res.annual_purchase_cost:
                lines.append(f"- **Total Annual Cost (including purchase)**: **${res.total_annual_cost:,.2f}**")
            if inputs.get("demand_conversion_note"):
                lines.append(f"- **Demand Unit Conversion**: {inputs['demand_conversion_note']}")
            if inputs.get("holding_conversion_note"):
                lines.append(f"- **Holding Cost Conversion**: {inputs['holding_conversion_note']}")
            lines.append("")
            lines.append("### 1. Cost Breakdown & Economic Balance")
            lines.append(f"- **Annual Ordering Cost**: ${res.annual_ordering_cost:,.2f}  [Formula: (D / Q) × S = ({res.annual_demand:,.0f} / {res.optimal_order_quantity:.2f}) × ${res.order_cost:.2f}]")
            lines.append(f"- **Annual Holding Cost**: ${res.annual_holding_cost:,.2f}  [Formula: (Q / 2) × H = ({res.optimal_order_quantity:.2f} / 2) × ${res.holding_cost_per_unit:.2f}]")
            lines.append(f"- **Cost Balance Delta**: ${res.cost_difference_holding_vs_ordering:.4f} (Optimal balance achieved where holding cost equals ordering cost)")
            lines.append("")
            lines.append("### 2. Operational Insights")
            lines.append(f"- Ordering in batches of {res.recommended_order_quantity:,} minimizes the sum of setup and inventory carrying costs.")
            lines.append(f"- If demand increases by 20%, EOQ scales sub-linearly by √1.20 (~9.5%), demonstrating economies of scale.")

        elif model_key == "quantity_discount":
            res: tools.QuantityDiscountResult = result
            best = res.best_tier
            lines.append("## Executive Summary: Quantity Discount Recommendation")
            lines.append(f"- **Optimal Order Quantity**: **{res.recommended_order_quantity:,} units** (Tier {best.tier_number})")
            lines.append(f"- **Negotiated Unit Price**: **${best.unit_price:.2f} per unit**")
            lines.append(f"- **Total Annual Cost**: **${res.total_annual_cost:,.2f}**")
            if res.cost_savings_vs_base > 0:
                lines.append(f"- **Annual Cost Savings vs Base Tier**: **${res.cost_savings_vs_base:,.2f}**")
            if inputs.get("demand_conversion_note"):
                lines.append(f"- **Demand Unit Conversion**: {inputs['demand_conversion_note']}")
            if inputs.get("holding_conversion_note"):
                lines.append(f"- **Holding Cost Conversion**: {inputs['holding_conversion_note']}")
            lines.append("")
            lines.append("### 1. Tier Evaluation Comparison Table")
            lines.append("| Tier | Range (Units) | Unit Price | Unconstrained EOQ | Candidate Q | Feasible? | Total Annual Cost |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in res.tiers_evaluated:
                max_str = f"{t.max_quantity:,.0f}" if t.max_quantity is not None else "∞"
                range_str = f"{t.min_quantity:,.0f} - {max_str}"
                feas_str = "Yes" if t.is_eoq_feasible else "Adjusted to break"
                best_marker = " ⭐ **BEST**" if t.tier_number == best.tier_number else ""
                lines.append(
                    f"| Tier {t.tier_number} | {range_str} | ${t.unit_price:.2f} | {t.eoq_unconstrained:.1f} | "
                    f"{t.candidate_quantity:,.0f} | {feas_str} | ${t.total_annual_cost:,.2f}{best_marker} |"
                )
            lines.append("")
            lines.append("### 2. Managerial Trade-off Analysis")
            lines.append(
                f"- Moving to Tier {best.tier_number} increases holding stock ({best.candidate_quantity:,.0f} units), "
                f"but the bulk purchase discount more than offsets carrying costs."
            )

        elif model_key == "safety_stock_rop":
            res: tools.SafetyStockROPResult = result
            lines.append("## Executive Summary: Safety Stock & Reorder Point Recommendation")
            lines.append(f"- **Recommended Safety Stock (SS)**: **{res.recommended_safety_stock:,} units** ({res.safety_stock:.2f} exact)")
            lines.append(f"- **Recommended Reorder Point (ROP)**: **{res.recommended_reorder_point:,} units** ({res.reorder_point:.2f} exact)")
            lines.append(f"- **Target Cycle Service Level**: **{res.service_level * 100:.1f}%** (Stockout risk: {res.stockout_risk * 100:.1f}%)")
            if inputs.get("conversion_note"):
                lines.append(f"- **Lead Time Unit Conversion**: {inputs['conversion_note']}")
            lines.append("")
            lines.append(f"### 1. Statistical Breakdown ({res.case_description})")
            lines.append(f"- **Normal Z-Score**: {res.z_score:.4f} standard deviations")
            lines.append(f"- **Expected Lead Time Demand (μ_L)**: {res.lead_time_demand_mean:,.2f} units [Formula: d × L = {res.daily_demand_mean:g} × {res.lead_time_days:g}]")
            lines.append(f"- **Lead Time Demand Std Dev (σ_LT)**: {res.lead_time_demand_std:.2f} units")
            if res.has_variable_lead_time:
                lines.append(f"- *Note*: Accounts for lead time uncertainty with lead time std dev = {res.lead_time_std_days:g} days.")
                lines.append(f"- **Demand Variance Component**: L × σ_d² = {res.lead_time_days:g} × {res.daily_demand_std:g}² = {res.demand_variance_component:,.2f}")
                lines.append(f"- **Lead Time Variance Component**: d² × σ_L² = {res.daily_demand_mean:g}² × {res.lead_time_std_days:g}² = {res.lead_time_variance_component:,.2f}")
            lines.append(f"- **Safety Stock Calculation**: SS = Z × σ_LT = {res.z_score:.4f} × {res.lead_time_demand_std:.2f} = {res.safety_stock:.2f} units")
            lines.append(f"- **Reorder Point Calculation**: ROP = μ_L + SS = {res.lead_time_demand_mean:.2f} + {res.safety_stock:.2f} = {res.reorder_point:.2f} units")
            lines.append("")
            lines.append("### 2. Operational Rule")
            lines.append(
                f"When inventory on-hand plus on-order drops to or below **{res.recommended_reorder_point:,} units**, "
                f"immediately place a replenishment order."
            )

        elif model_key == "fixed_period_review":
            res: tools.FixedPeriodReviewResult = result
            lines.append("## Executive Summary: Fixed-Period Review Order-Up-To Recommendation")
            lines.append(f"- **Target Order-Up-To Level (S)**: **{res.recommended_order_up_to_level:,} units** ({res.order_up_to_level:.2f} exact)")
            lines.append(f"- **Current Recommended Order Quantity (Q)**: **{res.recommended_order_quantity:,} units** ({res.order_quantity:.2f} exact)")
            lines.append(f"- **Safety Stock Buffer**: **{res.recommended_safety_stock:,} units**")
            lines.append("")
            lines.append("### 1. Protection Interval & Inventory Position")
            lines.append(f"- **Protection Interval (P + L)**: {res.protection_interval_days:.1f} days (Review Period: {res.review_period_days} days + Lead Time: {res.lead_time_days} days)")
            lines.append(f"- **Demand over Protection Interval**: {res.protection_demand_mean:,.1f} units (std dev: {res.protection_demand_std:.2f} units)")
            lines.append(f"- **Current Inventory Position (IP)**: {res.inventory_position:,.0f} units")
            lines.append(f"  - On-Hand: {res.current_on_hand:,.0f} units")
            lines.append(f"  - On-Order (pipeline): {res.current_on_order:,.0f} units")
            lines.append(f"  - Backorders: {res.current_backorders:,.0f} units")
            lines.append(f"- **Order Quantity Formula**: Q = max(0, S - IP) = max(0, {res.order_up_to_level:.1f} - {res.inventory_position:.1f}) = {res.order_quantity:.2f} units")
            lines.append("")
            lines.append("### 2. Operational Rule")
            lines.append(f"Review stock every {res.review_period_days:.0f} days and place an order to bring total inventory position up to {res.recommended_order_up_to_level:,} units.")

        elif model_key == "newsvendor":
            res: tools.NewsvendorResult = result
            lines.append("## Executive Summary: Newsvendor (Single-Period) Production Recommendation")
            lines.append(f"- **Optimal Order / Production Quantity (Q*)**: **{res.recommended_quantity:,} units** ({res.optimal_quantity:.2f} exact)")
            lines.append(f"- **Critical Fractile / Service Level (CR)**: **{res.critical_ratio * 100:.2f}%**")
            lines.append(f"- **Expected Total Profit**: **${res.expected_profit:,.2f}**")
            lines.append("")
            lines.append("### 1. Marginal Economic Cost Breakdown")
            lines.append(f"- **Cost of Underage (Cu - Lost Margin per unit)**: ${res.underage_cost:.2f}  [Price - Cost + Penalty = ${res.unit_price:.2f} - ${res.unit_cost:.2f} + ${res.shortage_penalty:.2f}]")
            lines.append(f"- **Cost of Overage (Co - Loss on Leftovers)**: ${res.overage_cost:.2f}  [Cost - Salvage = ${res.unit_cost:.2f} - ${res.salvage_value:.2f}]")
            lines.append(f"- **Critical Ratio**: Cu / (Cu + Co) = {res.underage_cost:.2f} / ({res.underage_cost:.2f} + {res.overage_cost:.2f}) = {res.critical_ratio:.4f}")
            lines.append(f"- **Normal Quantile (Z-score)**: {res.z_score:.4f}")
            lines.append("")
            lines.append("### 2. Expected Performance Payoffs")
            lines.append(f"- **Expected Sales**: {res.expected_sales:,.1f} units (Expected revenue: ${res.expected_revenue:,.2f})")
            lines.append(f"- **Expected Leftover Inventory**: {res.expected_leftover_units:,.1f} units (Salvage recovery: ${res.expected_salvage_recovery:,.2f})")
            lines.append(f"- **Expected Lost Sales (stockouts)**: {res.expected_lost_sales:,.1f} units")
            lines.append(f"- **In-Stock Probability**: {res.in_stock_probability * 100:.1f}%")

        # Verification audit section
        ver = result.verification if hasattr(result, "verification") else None
        if ver:
            lines.append("")
            lines.append("### 3. Verification & Operational Integrity Checks")
            for chk in ver.checks_passed:
                lines.append(f"- [x] Verified: {chk}")
            for w in ver.warnings:
                lines.append(f"- [!] Warning: {w}")

        # Complete 8-Step Student Verification Section (Deterministic Tool Output)
        if hasattr(result, "inputs_extracted"):
            lines.append("")
            lines.append("---")
            lines.append("### 🎓 Step-by-Step Verifiable Solution (Student Audit)")
            lines.append("All numerical calculations below are executed deterministically by the Python calculation tool.")
            lines.append("")
            
            # 1. Inputs Extracted
            lines.append("#### 1. Inputs Extracted")
            for k, v in result.inputs_extracted.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:g}")
                elif isinstance(v, list):
                    lines.append(f"- **{k}**: {len(v)} entries {v}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

            # 2. Variable Definitions
            lines.append("#### 2. Variable Definitions")
            for sym, defn in getattr(result, "variable_definitions", {}).items():
                lines.append(f"- `${sym}$`: {defn}")
            lines.append("")

            # 3. Formula
            lines.append("#### 3. Formula")
            lines.append(f"```text\n{getattr(result, 'formula', '')}\n```")
            lines.append("")

            # 4. Formula with Substituted Values
            lines.append("#### 4. Formula with Substituted Values")
            lines.append(f"```text\n{getattr(result, 'formula_substituted', '')}\n```")
            lines.append("")

            # 5. Intermediate Calculations
            lines.append("#### 5. Intermediate Calculations")
            for ik, iv in getattr(result, "intermediate_calculations", {}).items():
                if ik == "tier_step_by_step":
                    continue  # Table already provided in tier comparison
                if iv is None:
                    continue  # Skip uncalculated/unspecified parameters (e.g. annual_purchase_cost when C is omitted)
                if isinstance(iv, float):
                    lines.append(f"- **{ik}**: {iv:,.4f}")
                else:
                    lines.append(f"- **{ik}**: {iv}")
            lines.append("")

            # 6. Final Answer
            lines.append("#### 6. Final Answer")
            lines.append(f"**Recommended Decision**: **{getattr(result, 'final_answer', result.recommended_order_quantity if hasattr(result, 'recommended_order_quantity') else '')} units**")
            lines.append("")

            # 7. Verification Checks
            lines.append("#### 7. Verification Checks")
            for chk in getattr(result, "verification_checks", []):
                lines.append(f"- [x] {chk}")
            lines.append("")

            # 8. Assumptions
            lines.append("#### 8. Assumptions")
            for a in getattr(result, "assumptions", []):
                lines.append(f"- {a}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Agent Orchestrator
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """
    Main Agent Orchestrator executing the complete 7-step pipeline:
    Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
    """

    def __init__(self):
        self.identifier = ModelIdentifier()
        self.extractor = InputExtractor()
        self.selector = ToolSelector()
        self.explainer = Explainer()

    def ask(self, question: str) -> AgentResponse:
        pipeline: List[PipelineStep] = []

        # Step 1: Customer Question
        pipeline.append(PipelineStep(
            name="1. Customer Question",
            status="RECEIVED",
            details={"raw_question": question},
            summary=f"Received query: '{question[:80]}...'"
        ))

        # Step 2: Identify Model
        model_key, confidence, matched_kws = self.identifier.identify(question)
        model_name = self.identifier.MODEL_DISPLAY_NAMES.get(model_key, model_key)
        pipeline.append(PipelineStep(
            name="2. Identify Model",
            status="SUCCESS",
            details={"model_key": model_key, "model_name": model_name, "confidence": confidence, "matched_keywords": matched_kws},
            summary=f"Identified '{model_name}' (Confidence: {confidence*100:.0f}%)"
        ))

        # Step 3: Extract Inputs
        extracted_inputs, missing = self.extractor.extract_inputs(model_key, question)
        if missing:
            pipeline.append(PipelineStep(
                name="3. Extract Inputs",
                status="INCOMPLETE",
                details={"extracted": extracted_inputs, "missing": missing},
                summary=f"Extracted {len(extracted_inputs)} parameters; missing: {', '.join(missing)}"
            ))
            missing_msg = (
                f"### Additional Inputs Required for {model_name}\n"
                f"I identified that your question relates to **{model_name}**, but need the following parameters to calculate exact numerical recommendations:\n\n"
                + "\n".join([f"- **{m}**" for m in missing]) + "\n\n"
                f"Please provide these values, or provide a complete problem statement."
            )
            return AgentResponse(
                question=question,
                identified_model=model_key,
                model_display_name=model_name,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing,
                selected_tool=self.selector.TOOL_NAMES.get(model_key, ""),
                raw_result=None,
                verification=None,
                explanation=missing_msg,
                pipeline_trace=pipeline,
                success=False,
                error_message=f"Missing inputs: {', '.join(missing)}"
            )

        pipeline.append(PipelineStep(
            name="3. Extract Inputs",
            status="SUCCESS",
            details={"extracted_inputs": extracted_inputs},
            summary=f"Extracted {len(extracted_inputs)} parameters cleanly"
        ))

        # Step 4: Select Python Tool
        tool_name = self.selector.TOOL_NAMES.get(model_key, "unknown")
        pipeline.append(PipelineStep(
            name="4. Select Python Tool",
            status="SELECTED",
            details={"tool_name": tool_name},
            summary=f"Selected tool '{tool_name}'"
        ))

        # Step 5: Calculate
        try:
            executed_tool, raw_result = self.selector.execute(model_key, extracted_inputs)
            pipeline.append(PipelineStep(
                name="5. Calculate",
                status="SUCCESS",
                details={"tool": executed_tool},
                summary="Completed numerical operations with zero errors"
            ))
        except Exception as e:
            pipeline.append(PipelineStep(
                name="5. Calculate",
                status="FAILED",
                details={"error": str(e)},
                summary=f"Calculation failed: {str(e)}"
            ))
            return AgentResponse(
                question=question,
                identified_model=model_key,
                model_display_name=model_name,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool=tool_name,
                raw_result=None,
                verification=None,
                explanation=f"### Error During Calculation\nFailed to compute {model_name}: {str(e)}",
                pipeline_trace=pipeline,
                success=False,
                error_message=str(e)
            )

        # Step 6: Verify
        verification = getattr(raw_result, "verification", None)
        if verification and not verification.is_valid:
            pipeline.append(PipelineStep(
                name="6. Verify",
                status="FAILED",
                details={"errors": verification.errors, "warnings": verification.warnings},
                summary=f"Verification checks failed: {'; '.join(verification.errors)}"
            ))
            return AgentResponse(
                question=question,
                identified_model=model_key,
                model_display_name=model_name,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool=tool_name,
                raw_result=raw_result,
                verification=verification,
                explanation=f"### Result Verification Failed\n{'; '.join(verification.errors)}",
                pipeline_trace=pipeline,
                success=False,
                error_message="; ".join(verification.errors)
            )

        pipeline.append(PipelineStep(
            name="6. Verify",
            status="SUCCESS",
            details={"checks_passed": verification.checks_passed if verification else []},
            summary=f"Passed {len(verification.checks_passed) if verification else 0} operational verification checks"
        ))

        # Step 7: Explain
        explanation = self.explainer.explain(model_key, raw_result, extracted_inputs)
        pipeline.append(PipelineStep(
            name="7. Explain",
            status="SUCCESS",
            details={},
            summary="Synthesized executive managerial explanation with numerical breakdowns"
        ))

        return AgentResponse(
            question=question,
            identified_model=model_key,
            model_display_name=model_name,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool=tool_name,
            raw_result=raw_result,
            verification=verification,
            explanation=explanation,
            pipeline_trace=pipeline,
            success=True
        )

    def ask_structured(self, model_key: str, inputs: Dict[str, Any]) -> AgentResponse:
        """Direct execution for structured inputs or programmatic calls."""
        model_name = self.identifier.MODEL_DISPLAY_NAMES.get(model_key, model_key)
        tool_name, raw_result = self.selector.execute(model_key, inputs)
        ver = getattr(raw_result, "verification", None)
        explanation = self.explainer.explain(model_key, raw_result, inputs)
        return AgentResponse(
            question="[Structured API Call]",
            identified_model=model_key,
            model_display_name=model_name,
            extracted_inputs=inputs,
            missing_inputs=[],
            selected_tool=tool_name,
            raw_result=raw_result,
            verification=ver,
            explanation=explanation,
            success=True
        )
