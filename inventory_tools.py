"""
Supply Chain Analytics Deterministic Inventory Tools
====================================================
Deterministic, validated operations research calculation tools:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- Validates inputs strictly (raises ValueError for invalid or missing inputs)
- Returns structured results with dataclasses
- Returns detailed intermediate calculations
- Returns the final answer
- Identifies underlying operations research assumptions
- Never invents missing numerical inputs
- Returns all 8 verifiable calculation components:
  1. Inputs Extracted
  2. Variable Definitions
  3. Formula
  4. Formula with Substituted Values
  5. Intermediate Calculations
  6. Final Answer
  7. Verification Checks
  8. Assumptions
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union


# ---------------------------------------------------------------------------
# Verification & Base Data Models
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistical Helper Functions
# ---------------------------------------------------------------------------

def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Cumulative distribution function for normal distribution."""
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Probability density function for normal distribution."""
    if std <= 0:
        return 0.0
    z = (x - mean) / std
    return (1.0 / (std * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)


def normal_inv_cdf(p: float, mean: float = 0.0, std: float = 1.0) -> float:
    """
    Inverse cumulative distribution function (Quantile function / PPF).
    Uses statistics.NormalDist for high precision.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be strictly between 0 and 1, got {p}")
    dist = statistics.NormalDist(mu=mean, sigma=std)
    return dist.inv_cdf(p)


def standard_normal_loss_function(z: float) -> float:
    """
    Standard normal loss function: L(z) = phi(z) - z * (1 - Phi(z))
    Represents expected shortage units for standard normal N(0, 1).
    """
    phi_z = normal_pdf(z, 0.0, 1.0)
    phi_upper = 1.0 - normal_cdf(z, 0.0, 1.0)
    return phi_z - z * phi_upper


def parse_service_level(service_level: Optional[float]) -> float:
    """
    Normalizes service level to decimal in (0, 1).
    Accepts 0.95 or 95.0. Rejects ambiguous or out-of-bounds values.
    Never invents a default if service_level is None.
    """
    if service_level is None:
        raise ValueError("Service level is required and cannot be None.")
    if service_level <= 0.0:
        raise ValueError(f"Service level must be > 0, got {service_level}")
    if service_level > 1.0:
        if service_level < 10.0:
            raise ValueError(f"Ambiguous service level: {service_level}. Provide decimal (e.g. 0.95) or percentage (e.g. 95).")
        elif service_level <= 100.0:
            return service_level / 100.0
        else:
            raise ValueError(f"Service level percentage cannot exceed 100%, got {service_level}%")
    return service_level


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

@dataclass
class EOQResult:
    final_answer: int
    optimal_order_quantity: float
    recommended_order_quantity: int
    annual_demand: float
    order_cost: float
    holding_cost_per_unit: float
    unit_purchase_cost: Optional[float]
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_inventory_cost: float
    annual_purchase_cost: Optional[float]
    total_annual_cost: float
    cost_difference_holding_vs_ordering: float
    working_days: int
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_substituted: str
    verification_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_verification_details(self) -> Dict[str, Any]:
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_substituted": self.formula_substituted,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    working_days: Optional[int] = 250,
) -> EOQResult:
    """
    Calculates deterministic Economic Order Quantity (EOQ):
        Q* = sqrt((2 * D * S) / H)
        
    Strict Validation:
        - demand (D) > 0
        - order_cost (S) > 0
        - Either holding_cost (H) > 0, OR both unit_cost (C) > 0 and holding_rate (i) > 0
        - Never invents missing numerical inputs
    """
    # 1. Validate inputs (no hallucinating missing numbers)
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (Annual Demand D).")
    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (Order Setup Cost S).")
    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0), got {demand}.")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be strictly positive (> 0), got {order_cost}.")
    if working_days is None or working_days <= 0:
        raise ValueError(f"Working days must be strictly positive (> 0), got {working_days}.")

    # Resolve holding cost
    if holding_cost is None:
        if unit_cost is None or holding_rate is None:
            raise ValueError(
                "Missing required holding cost input: Provide either 'holding_cost' (H) "
                "or both 'unit_cost' (C) and 'holding_rate' (i)."
            )
        if unit_cost <= 0:
            raise ValueError(f"Unit cost must be strictly positive (> 0), got {unit_cost}.")
        if holding_rate <= 0:
            raise ValueError(f"Holding rate must be strictly positive (> 0), got {holding_rate}.")
        rate_decimal = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
        h = rate_decimal * unit_cost
        h_method = f"Derived: holding_rate ({rate_decimal*100:.1f}%) * unit_cost (${unit_cost:.2f})"
    else:
        if holding_cost <= 0:
            raise ValueError(f"Holding cost per unit must be strictly positive (> 0), got {holding_cost}.")
        h = holding_cost
        h_method = f"Direct: ${holding_cost:.2f} per unit/year"

    # 2. Intermediate calculations
    numerator = 2.0 * demand * order_cost
    ratio = numerator / h
    q_opt = math.sqrt(ratio)
    q_rec = int(math.ceil(q_opt))

    orders_per_year = demand / q_opt
    cycle_time_years = q_opt / demand
    cycle_time_days = cycle_time_years * working_days

    annual_ordering_cost = (demand / q_opt) * order_cost
    annual_holding_cost = (q_opt / 2.0) * h
    annual_inventory_cost = annual_ordering_cost + annual_holding_cost
    cost_diff = abs(annual_holding_cost - annual_ordering_cost)

    annual_purchase_cost = (demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = annual_inventory_cost + (annual_purchase_cost or 0.0)

    intermediates = {
        "holding_cost_method": h_method,
        "holding_cost_used": h,
        "eoq_numerator (2*D*S)": numerator,
        "eoq_ratio (2*D*S/H)": ratio,
        "exact_optimal_q": q_opt,
        "orders_per_year (D/Q)": orders_per_year,
        "cycle_time_days": cycle_time_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "cost_balance_delta": cost_diff,
        "annual_inventory_cost (Hold + Order)": annual_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
    }

    # 3. Operations Research Assumptions
    assumptions = [
        "Annual demand is deterministic, constant, and continuous over the planning horizon.",
        "Order replenishment lead time is instantaneous (or constant with orders arriving in single batches).",
        "Order setup cost (S) is fixed and independent of the order size.",
        "Inventory holding cost (H) is linear and proportional to the average inventory held.",
        "No shortages, stockouts, or backorders are permitted.",
        "Unit purchase price is constant and independent of order quantity (no discounts).",
        "The inventory horizon is infinite and working days are evenly distributed.",
    ]

    # 4. Verification checks
    checks_passed = [
        f"Strict positivity: Demand D={demand:g} > 0 and Order Cost S={order_cost:g} > 0 verified.",
        f"Holding cost resolution: Valid H=${h:.2f} > 0 verified ({h_method}).",
        f"Economic balance condition: Annual holding (${annual_holding_cost:.2f}) equals annual ordering (${annual_ordering_cost:.2f}) with delta ${cost_diff:.4f}.",
        f"Convexity / Second-Order Condition: d²TC/dQ² = 2DS/Q³ = {2.0 * demand * order_cost / (q_opt ** 3):.6f} > 0 (confirms strict cost minimum).",
    ]
    verification = VerificationResult(
        is_valid=True,
        checks_passed=checks_passed,
    )

    # 5. Verifiable Step Components
    raw_inputs = {
        "Annual Demand (D)": demand,
        "Order/Setup Cost (S)": order_cost,
        "Holding Cost per Unit (H)": h,
        "Unit Purchase Cost (C)": unit_cost,
        "Holding Cost Rate (i)": holding_rate,
        "Working Days per Year (W)": working_days,
    }
    inputs_extracted = {k: v for k, v in raw_inputs.items() if v is not None}

    variable_definitions = {
        "D": "Annual demand (units per year)",
        "S": "Ordering or setup cost incurred per replenishment order ($/order)",
        "H": "Holding (carrying) cost per unit of inventory held per year ($/unit/year)",
        "C": "Unit purchase price ($/unit)",
        "i": "Annual inventory holding cost rate (fraction/year)",
        "Q*": "Optimal Economic Order Quantity (units)",
        "TC": "Total annual inventory cost ($/year)",
    }

    formula = "Q* = sqrt((2 * D * S) / H)"
    formula_substituted = (
        f"Q* = sqrt((2 * {demand:g} * {order_cost:g}) / {h:g}) = "
        f"sqrt({numerator:g} / {h:g}) = sqrt({ratio:.4f}) = {q_opt:.4f}"
    )

    return EOQResult(
        final_answer=q_rec,
        optimal_order_quantity=q_opt,
        recommended_order_quantity=q_rec,
        annual_demand=demand,
        order_cost=order_cost,
        holding_cost_per_unit=h,
        unit_purchase_cost=unit_cost,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_inventory_cost=annual_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        cost_difference_holding_vs_ordering=cost_diff,
        working_days=working_days,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        verification=verification,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_substituted=formula_substituted,
        verification_checks=checks_passed,
    )


# ---------------------------------------------------------------------------
# 2. Quantity Discount Model
# ---------------------------------------------------------------------------

@dataclass
class DiscountTier:
    tier_number: int
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    eoq_unconstrained: float
    candidate_quantity: float
    is_eoq_feasible: bool
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    final_answer: int
    optimal_order_quantity: float
    recommended_order_quantity: int
    best_tier: DiscountTier
    annual_demand: float
    order_cost: float
    holding_cost_mode: str
    tiers_evaluated: List[DiscountTier]
    total_annual_cost: float
    cost_savings_vs_base: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_substituted: str
    verification_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_verification_details(self) -> Dict[str, Any]:
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_substituted": self.formula_substituted,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_tiers: Optional[List[Tuple[float, Optional[float], float]]] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluates quantity discount price breaks to determine the global cost-minimizing order quantity.
    
    Supports:
        - Fixed holding cost (holding_cost in $/unit/year)
        - Percentage holding cost (holding_rate as % or decimal of unit price: H_j = i * C_j)
        
    Strict Validation:
        - demand > 0
        - order_cost > 0
        - discount_tiers non-empty, sorted, valid intervals
        - Either holding_cost > 0 or holding_rate > 0
        - Never invents missing numerical inputs
    """
    # 1. Validation
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (Annual Demand D).")
    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (Order Setup Cost S).")
    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0), got {demand}.")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be strictly positive (> 0), got {order_cost}.")
    if not discount_tiers:
        raise ValueError("Missing required input: 'discount_tiers' cannot be None or empty.")

    if holding_cost is None and holding_rate is None:
        raise ValueError("Must provide either 'holding_cost' (fixed $/unit) or 'holding_rate' (percentage of price).")

    # Validate holding cost mode
    if holding_rate is not None:
        if holding_rate <= 0:
            raise ValueError(f"Holding rate must be strictly positive (> 0), got {holding_rate}.")
        rate_dec = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
        holding_mode = f"Percentage holding cost ({rate_dec * 100:.1f}% of tier price)"
    else:
        if holding_cost is not None and holding_cost <= 0:
            raise ValueError(f"Holding cost must be strictly positive (> 0), got {holding_cost}.")
        rate_dec = None
        holding_mode = f"Fixed holding cost (${holding_cost:.2f}/unit/year)"

    # Validate tiers
    sorted_tiers = sorted(discount_tiers, key=lambda t: t[0])
    for min_q, max_q, pr in sorted_tiers:
        if min_q < 0:
            raise ValueError(f"Tier min quantity must be >= 0, got {min_q}.")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"Tier max quantity ({max_q}) cannot be less than min ({min_q}).")
        if pr <= 0:
            raise ValueError(f"Tier unit price must be strictly positive (> 0), got {pr}.")

    # 2. Evaluate all tiers
    evaluated: List[DiscountTier] = []
    tier_intermediates: List[Dict[str, Any]] = []

    for i, (min_q, max_q, price) in enumerate(sorted_tiers, start=1):
        tier_h = (rate_dec * price) if rate_dec is not None else float(holding_cost)
        eoq_raw = math.sqrt((2.0 * demand * order_cost) / tier_h)

        # Feasibility check
        is_feasible = (min_q <= eoq_raw and (max_q is None or eoq_raw <= max_q))

        if is_feasible:
            candidate_q = eoq_raw
            feasibility_reason = "EOQ falls within tier boundary; optimal for this tier."
        elif eoq_raw < min_q:
            candidate_q = float(min_q)
            feasibility_reason = f"EOQ ({eoq_raw:.1f}) < min ({min_q}); adjusted upward to price-break threshold."
        else:
            candidate_q = float(max_q) if max_q is not None else eoq_raw
            feasibility_reason = f"EOQ ({eoq_raw:.1f}) > max ({max_q}); capped at tier max."

        purch_c = demand * price
        ord_c = (demand / candidate_q) * order_cost
        h_c = (candidate_q / 2.0) * tier_h
        tot_c = purch_c + ord_c + h_c

        tier_obj = DiscountTier(
            tier_number=i,
            min_quantity=min_q,
            max_quantity=max_q,
            unit_price=price,
            holding_cost_per_unit=tier_h,
            eoq_unconstrained=eoq_raw,
            candidate_quantity=candidate_q,
            is_eoq_feasible=is_feasible,
            annual_purchase_cost=purch_c,
            annual_ordering_cost=ord_c,
            annual_holding_cost=h_c,
            total_annual_cost=tot_c,
        )
        evaluated.append(tier_obj)
        tier_intermediates.append({
            "tier": i,
            "price": price,
            "holding_cost": tier_h,
            "unconstrained_eoq": eoq_raw,
            "is_feasible": is_feasible,
            "candidate_quantity": candidate_q,
            "feasibility_action": feasibility_reason,
            "purchase_cost": purch_c,
            "ordering_cost": ord_c,
            "holding_cost": h_c,
            "total_cost": tot_c,
        })

    best_tier = min(evaluated, key=lambda t: t.total_annual_cost)
    base_tier = evaluated[0]
    savings = max(0.0, base_tier.total_annual_cost - best_tier.total_annual_cost)
    q_rec = int(math.ceil(best_tier.candidate_quantity))

    intermediates = {
        "holding_cost_mode": holding_mode,
        "tiers_evaluated_count": len(evaluated),
        "tier_step_by_step": tier_intermediates,
        "selected_optimal_tier": best_tier.tier_number,
        "selected_candidate_quantity": best_tier.candidate_quantity,
        "total_annual_cost": best_tier.total_annual_cost,
        "cost_savings_vs_base_tier": savings,
    }

    assumptions = [
        "All-units quantity discount structure applies (all units ordered receive the discount price).",
        "Annual demand is deterministic and constant across all price schedules.",
        "Supplier has unlimited production capacity to fulfill any tier threshold immediately.",
        "Holding cost is either constant across all tiers or scales strictly with the unit purchase price.",
        "Order replenishment occurs in full single batches with zero internal stockout tolerance.",
    ]

    checks_passed = [
        f"Evaluated {len(evaluated)} discount tiers.",
        f"Determined optimal Tier {best_tier.tier_number} at ${best_tier.unit_price:.2f}/unit.",
        f"Candidate quantity Q={best_tier.candidate_quantity:g} satisfies tier range [{best_tier.min_quantity:g}, {best_tier.max_quantity if best_tier.max_quantity is not None else 'inf'}).",
        f"Annual cost minimized at ${best_tier.total_annual_cost:,.2f} with savings of ${savings:,.2f} vs base tier.",
    ]
    verification = VerificationResult(
        is_valid=True,
        checks_passed=checks_passed,
    )

    raw_inputs = {
        "Annual Demand (D)": demand,
        "Order/Setup Cost (S)": order_cost,
        "Discount Tiers": discount_tiers,
        "Fixed Holding Cost (H)": holding_cost,
        "Holding Cost Rate (i)": holding_rate,
    }
    inputs_extracted = {k: v for k, v in raw_inputs.items() if v is not None}

    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering or setup cost per order ($/order)",
        "C_j": "Unit purchase price for tier j ($/unit)",
        "H_j": "Holding cost per unit for tier j ($/unit/year)",
        "q_j": "Minimum quantity threshold / price break for tier j (units)",
        "Q_j*": "Unconstrained EOQ for tier j = sqrt((2 * D * S) / H_j)",
        "Candidate Q_j": "Feasible evaluated order quantity for tier j",
        "TC_j": "Total annual cost = D * C_j + (D / Q_j) * S + (Q_j / 2) * H_j",
    }

    formula = (
        "For each tier j:\n"
        "1. Q_j* = sqrt((2 * D * S) / H_j)\n"
        "2. Candidate Q_j = Q_j* if (min_j <= Q_j* <= max_j) else min_j\n"
        "3. TC_j = D * C_j + (D / Candidate Q_j) * S + (Candidate Q_j / 2) * H_j\n"
        "Optimal Order Quantity = Candidate Q for tier with min(TC_j)"
    )

    formula_substituted = (
        f"Optimal Tier {best_tier.tier_number} (Price C = ${best_tier.unit_price:.2f}, H = ${best_tier.holding_cost_per_unit:.2f}):\n"
        f"Q* unconstrained = sqrt((2 * {demand:g} * {order_cost:g}) / {best_tier.holding_cost_per_unit:.4f}) = {best_tier.eoq_unconstrained:.2f}\n"
        f"Candidate Q = {best_tier.candidate_quantity:.2f} ({'feasible' if best_tier.is_eoq_feasible else 'adjusted to price break'})\n"
        f"TC = ({demand:g} * ${best_tier.unit_price:.2f}) + ({demand:g} / {best_tier.candidate_quantity:.2f}) * ${order_cost:g} + "
        f"({best_tier.candidate_quantity:.2f} / 2) * ${best_tier.holding_cost_per_unit:.2f} = ${best_tier.total_annual_cost:,.2f}"
    )

    return QuantityDiscountResult(
        final_answer=q_rec,
        optimal_order_quantity=best_tier.candidate_quantity,
        recommended_order_quantity=q_rec,
        best_tier=best_tier,
        annual_demand=demand,
        order_cost=order_cost,
        holding_cost_mode=holding_mode,
        tiers_evaluated=evaluated,
        total_annual_cost=best_tier.total_annual_cost,
        cost_savings_vs_base=savings,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        verification=verification,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_substituted=formula_substituted,
        verification_checks=checks_passed,
    )


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point (Continuous Review)
# ---------------------------------------------------------------------------

@dataclass
class ROPResult:
    final_answer: int
    reorder_point: float
    recommended_reorder_point: int
    safety_stock: float
    recommended_safety_stock: int
    case_description: str
    daily_demand_mean: float
    daily_demand_std: float
    lead_time_days: float
    lead_time_std_days: float
    service_level: float
    z_score: float
    lead_time_demand_mean: float
    lead_time_demand_std: float
    demand_variance_component: float
    lead_time_variance_component: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_substituted: str
    verification_checks: List[str]
    stockout_risk: float = 0.05
    has_variable_lead_time: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_verification_details(self) -> Dict[str, Any]:
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_substituted": self.formula_substituted,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_rop(
    demand_mean: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_std: Optional[float] = 0.0,
    lead_time_std: Optional[float] = 0.0,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
) -> ROPResult:
    """
    Calculates deterministic Safety Stock and Reorder Point (ROP).
    
    Supports all 4 cases:
        1. Constant demand, constant lead time (demand_std=0, lead_time_std=0)
           sigma_LT = 0, SS = 0, ROP = d * L
        2. Variable demand, constant lead time (demand_std>0, lead_time_std=0)
           sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
        3. Constant demand, variable lead time (demand_std=0, lead_time_std>0)
           sigma_LT = d * sigma_L, SS = z * sigma_LT, ROP = d * L + SS
        4. Variable demand, variable lead time (demand_std>0, lead_time_std>0)
           sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
           SS = z * sigma_LT
           ROP = d * L + SS
           
    Strict Validation:
        - demand_mean (d) > 0
        - lead_time (L) > 0
        - demand_std >= 0, lead_time_std >= 0
        - Either service_level (in (0, 1) or percentage) OR z must be provided if variability exists
        - Never invents missing numerical inputs
    """
    # 1. Input validation
    if demand_mean is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (Average demand rate d).")
    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' (Supplier lead time L).")
    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be strictly positive (> 0), got {demand_mean}.")
    if lead_time <= 0:
        raise ValueError(f"Lead time must be strictly positive (> 0), got {lead_time}.")

    sigma_d = 0.0 if demand_std is None else float(demand_std)
    sigma_l = 0.0 if lead_time_std is None else float(lead_time_std)

    if sigma_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}.")
    if sigma_l < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative, got {sigma_l}.")

    is_demand_var = (sigma_d > 0.0)
    is_lt_var = (sigma_l > 0.0)
    has_variability = is_demand_var or is_lt_var

    # Resolve z and service level
    if has_variability:
        if z is not None:
            z_score = float(z)
            csl = normal_cdf(z_score)
        elif service_level is not None:
            csl = parse_service_level(service_level)
            z_score = normal_inv_cdf(csl)
        else:
            raise ValueError(
                "Variability detected: Must provide either 'service_level' (e.g. 0.95) "
                "or 'z' (standard normal quantile) to calculate safety stock."
            )
    else:
        # Deterministic: no safety stock required
        csl = 1.0
        z_score = 0.0

    # 2. Case execution
    mu_l = demand_mean * lead_time

    if not is_demand_var and not is_lt_var:
        case_name = "Case 1: Constant Demand, Constant Lead Time"
        var_demand = 0.0
        var_lt = 0.0
        sigma_lt = 0.0
        ss = 0.0
        formula = "ROP = d * L (SS = 0)"
        formula_substituted = f"ROP = {demand_mean:g} * {lead_time:g} = {mu_l:.2f}"
        case_assumptions = [
            "Demand rate and supplier lead time are strictly constant and deterministic.",
            "Zero safety stock is required because no stockout risk exists.",
        ]
    elif is_demand_var and not is_lt_var:
        case_name = "Case 2: Variable Demand, Constant Lead Time"
        var_demand = lead_time * (sigma_d ** 2)
        var_lt = 0.0
        sigma_lt = sigma_d * math.sqrt(lead_time)
        ss = z_score * sigma_lt
        formula = "sigma_LT = sigma_d * sqrt(L); SS = z * sigma_LT; ROP = d * L + SS"
        formula_substituted = (
            f"sigma_LT = {sigma_d:g} * sqrt({lead_time:g}) = {sigma_lt:.4f}; "
            f"SS = {z_score:.4f} * {sigma_lt:.4f} = {ss:.2f}; "
            f"ROP = ({demand_mean:g} * {lead_time:g}) + {ss:.2f} = {mu_l:.2f} + {ss:.2f} = {mu_l + ss:.2f}"
        )
        case_assumptions = [
            "Lead time is strictly constant; demand is independently and identically distributed.",
            "Demand during lead time follows a normal distribution N(d*L, sigma_d * sqrt(L)).",
        ]
    elif not is_demand_var and is_lt_var:
        case_name = "Case 3: Constant Demand, Variable Lead Time"
        var_demand = 0.0
        var_lt = (demand_mean ** 2) * (sigma_l ** 2)
        sigma_lt = demand_mean * sigma_l
        ss = z_score * sigma_lt
        formula = "sigma_LT = d * sigma_L; SS = z * sigma_LT; ROP = d * L + SS"
        formula_substituted = (
            f"sigma_LT = {demand_mean:g} * {sigma_l:g} = {sigma_lt:.4f}; "
            f"SS = {z_score:.4f} * {sigma_lt:.4f} = {ss:.2f}; "
            f"ROP = ({demand_mean:g} * {lead_time:g}) + {ss:.2f} = {mu_l:.2f} + {ss:.2f} = {mu_l + ss:.2f}"
        )
        case_assumptions = [
            "Demand rate is strictly constant; lead time is normally distributed with standard deviation sigma_L.",
            "Lead time variance translates to demand uncertainty via d * sigma_L.",
        ]
    else:
        case_name = "Case 4: Variable Demand, Variable Lead Time"
        var_demand = lead_time * (sigma_d ** 2)
        var_lt = (demand_mean ** 2) * (sigma_l ** 2)
        sigma_lt = math.sqrt(var_demand + var_lt)
        ss = z_score * sigma_lt
        formula = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2); SS = z * sigma_LT; ROP = d * L + SS"
        formula_substituted = (
            f"sigma_LT = sqrt({lead_time:g} * ({sigma_d:g})^2 + ({demand_mean:g})^2 * ({sigma_l:g})^2) = "
            f"sqrt({var_demand:.4f} + {var_lt:.4f}) = {sigma_lt:.4f}; "
            f"SS = {z_score:.4f} * {sigma_lt:.4f} = {ss:.2f}; "
            f"ROP = ({demand_mean:g} * {lead_time:g}) + {ss:.2f} = {mu_l:.2f} + {ss:.2f} = {mu_l + ss:.2f}"
        )
        case_assumptions = [
            "Both demand and lead time are stochastic and mutually independent.",
            "Total lead time demand variance equals L * sigma_d^2 + d^2 * sigma_L^2.",
            "Lead time demand distribution is well-approximated by a normal distribution.",
        ]

    rop = mu_l + ss
    rec_ss = max(0, int(math.ceil(ss)))
    rec_rop = max(0, int(math.ceil(rop)))

    intermediates = {
        "case_identified": case_name,
        "lead_time_demand_mean (d * L)": mu_l,
        "demand_variance_component (L * sigma_d^2)": var_demand,
        "lead_time_variance_component (d^2 * sigma_L^2)": var_lt,
        "total_lead_time_std_dev (sigma_LT)": sigma_lt,
        "z_score_used": z_score,
        "cycle_service_level": csl,
        "exact_safety_stock (SS)": ss,
        "exact_reorder_point (ROP)": rop,
    }

    all_assumptions = [
        "Continuous review policy (s, Q): inventory position is monitored continuously.",
        "Replenishment order of batch size Q is placed immediately when inventory position drops to or below ROP.",
        "Cycle service level represents the probability that demand during lead time does not exceed the reorder point.",
        "Unfilled customer demand during stockout is fully backordered.",
    ] + case_assumptions

    checks_passed = [
        f"Identified {case_name}.",
        f"Mean lead time demand = {mu_l:.2f} units.",
        f"Combined lead time std dev sigma_LT = {sigma_lt:.2f} units.",
        f"Safety stock = {rec_ss} units for CSL = {csl*100:.1f}% (Z={z_score:.4f}).",
        f"Reorder point = {rec_rop} units (ROP >= Mean lead time demand of {mu_l:.2f}).",
    ]
    verification = VerificationResult(
        is_valid=True,
        checks_passed=checks_passed,
    )

    raw_inputs = {
        "Average Demand (d)": demand_mean,
        "Supplier Lead Time (L)": lead_time,
        "Demand Std Dev (sigma_d)": sigma_d,
        "Lead Time Std Dev (sigma_L)": sigma_l,
        "Cycle Service Level (CSL)": csl if has_variability else None,
        "Z-Score (z)": z_score if has_variability else None,
    }
    inputs_extracted = {k: v for k, v in raw_inputs.items() if v is not None}

    variable_definitions = {
        "d": "Average demand rate per unit time",
        "L": "Average supplier lead time",
        "sigma_d": "Standard deviation of periodic demand",
        "sigma_L": "Standard deviation of lead time",
        "CSL": "Cycle Service Level (in-stock probability during lead time)",
        "Z": "Standard normal inverse CDF quantile for target CSL",
        "sigma_LT": "Combined standard deviation of demand during lead time",
        "SS": "Safety stock buffer = Z * sigma_LT",
        "ROP": "Reorder point = (d * L) + SS",
    }

    return ROPResult(
        final_answer=rec_rop,
        reorder_point=rop,
        recommended_reorder_point=rec_rop,
        safety_stock=ss,
        recommended_safety_stock=rec_ss,
        case_description=case_name,
        daily_demand_mean=demand_mean,
        daily_demand_std=sigma_d,
        lead_time_days=lead_time,
        lead_time_std_days=sigma_l,
        service_level=csl,
        z_score=z_score,
        lead_time_demand_mean=mu_l,
        lead_time_demand_std=sigma_lt,
        demand_variance_component=var_demand,
        lead_time_variance_component=var_lt,
        intermediate_calculations=intermediates,
        assumptions=all_assumptions,
        verification=verification,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_substituted=formula_substituted,
        verification_checks=checks_passed,
        stockout_risk=max(0.0, 1.0 - csl),
        has_variable_lead_time=is_lt_var,
    )


# Backward-compatible aliases
calculate_safety_stock_rop = calculate_rop
SafetyStockROPResult = ROPResult


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review Model (Periodic Review Order-Up-To Level)
# ---------------------------------------------------------------------------

@dataclass
class PeriodicReviewResult:
    final_answer: int
    order_quantity: float
    recommended_order_quantity: int
    order_up_to_level: float
    recommended_order_up_to_level: int
    safety_stock: float
    recommended_safety_stock: int
    review_period_days: float
    lead_time_days: float
    protection_period_days: float
    daily_demand_mean: float
    daily_demand_std: float
    protection_demand_mean: float
    protection_demand_std: float
    service_level: float
    z_score: float
    current_on_hand: float
    current_on_order: float
    current_backorders: float
    inventory_position: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_substituted: str
    verification_checks: List[str]
    protection_interval_days: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_verification_details(self) -> Dict[str, Any]:
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_substituted": self.formula_substituted,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    on_hand: Optional[float] = 0.0,
    on_order: Optional[float] = 0.0,
    backorders: Optional[float] = 0.0,
    inventory_position: Optional[float] = None,
    safety_stock: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Calculates deterministic Fixed-Period Review (Periodic Review Order-Up-To Level T / S):
        Protection period = review period (P) + lead time (L)
        Demand during protection period = d * (P + L)
        Safety stock = z * sigma_d * sqrt(P + L) (or explicit safety_stock)
        Target inventory = Demand during protection period + Safety stock
        Inventory position (IP) = on_hand + on_order - backorders (or explicit inventory_position)
        Order quantity = max(0, Target inventory - Inventory position)
        
    Strict Validation:
        - review_period (P) > 0
        - lead_time (L) >= 0
        - demand_mean (d) > 0
        - demand_std >= 0
        - Either service_level OR z must be provided if demand_std > 0 and safety_stock is None
        - Never invents missing numerical inputs
    """
    # 1. Input validation
    if review_period is None:
        raise ValueError("Missing required numerical input: 'review_period' (Review cycle interval P).")
    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' (Supplier lead time L).")
    if demand_mean is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (Average demand per period d).")

    if review_period <= 0:
        raise ValueError(f"Review period P must be strictly positive (> 0), got {review_period}.")
    if lead_time < 0:
        raise ValueError(f"Lead time L cannot be negative, got {lead_time}.")
    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be strictly positive (> 0), got {demand_mean}.")

    sigma_d = 0.0 if demand_std is None else float(demand_std)
    if sigma_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}.")

    # Resolve z and service level
    if safety_stock is not None:
        ss = float(safety_stock)
        z_score = 0.0
        csl = 1.0
    elif sigma_d > 0:
        if z is not None:
            z_score = float(z)
            csl = normal_cdf(z_score)
        elif service_level is not None:
            csl = parse_service_level(service_level)
            z_score = normal_inv_cdf(csl)
        else:
            raise ValueError(
                "Demand is stochastic (demand_std > 0): Must provide either 'service_level' "
                "or 'z' to calculate target inventory safety stock."
            )
        protection_period = review_period + lead_time
        prot_std = sigma_d * math.sqrt(protection_period)
        ss = z_score * prot_std
    else:
        csl = 1.0
        z_score = 0.0
        ss = 0.0

    oh = 0.0 if on_hand is None else float(on_hand)
    oo = 0.0 if on_order is None else float(on_order)
    bo = 0.0 if backorders is None else float(backorders)

    if oo < 0:
        raise ValueError(f"On-order pipeline cannot be negative, got {oo}.")
    if bo < 0:
        raise ValueError(f"Backorders cannot be negative, got {bo}.")

    # 2. Intermediate calculations
    protection_period = review_period + lead_time
    prot_mean = demand_mean * protection_period
    prot_std = sigma_d * math.sqrt(protection_period)
    if safety_stock is None and sigma_d > 0:
        ss = z_score * prot_std

    target_inventory = prot_mean + ss

    if inventory_position is not None:
        ip = float(inventory_position)
    else:
        ip = oh + oo - bo

    raw_order_qty = target_inventory - ip
    order_qty = max(0.0, raw_order_qty)

    rec_ss = max(0, int(math.ceil(ss)))
    rec_target = max(0, int(math.ceil(target_inventory)))
    rec_order = max(0, int(math.ceil(order_qty)))

    intermediates = {
        "protection_period (P + L)": protection_period,
        "expected_demand_during_protection (d * (P + L))": prot_mean,
        "std_dev_during_protection (sigma_d * sqrt(P + L))": prot_std,
        "z_score_used": z_score,
        "exact_safety_stock (SS)": ss,
        "target_inventory_level (S)": target_inventory,
        "current_inventory_position (IP = OH + OO - BO)": ip,
        "unconstrained_order_quantity (S - IP)": raw_order_qty,
        "constrained_order_quantity max(0, S - IP)": order_qty,
    }

    assumptions = [
        "Inventory is reviewed at discrete, fixed periodic intervals of length P.",
        "Replenishment order arrives after constant lead time L.",
        "Protection period must cover demand over interval P + L until the next replenishment arrives.",
        "Demand across periods is independent, stationary, and identically distributed.",
        "Target inventory (Order-Up-To Level) is sized to provide the designated cycle service level.",
        "Unfulfilled customer demand is completely backordered until stock arrives.",
    ]

    checks_passed = [
        f"Protection period = {protection_period:.1f} periods (Review {review_period:g} + Lead Time {lead_time:g}).",
        f"Target order-up-to inventory level = {rec_target} units (Target >= Expected protection demand of {prot_mean:.2f}).",
        f"Current inventory position = {ip:.1f} units (On-Hand: {oh:g}, On-Order: {oo:g}, Backorders: {bo:g}).",
        f"Recommended order quantity = {rec_order} units (Non-negative order quantity max(0, Target - IP)).",
    ]
    verification = VerificationResult(
        is_valid=True,
        checks_passed=checks_passed,
    )

    raw_inputs = {
        "Review Period (P)": review_period,
        "Lead Time (L)": lead_time,
        "Demand Mean (d)": demand_mean,
        "Demand Std Dev (sigma_d)": sigma_d,
        "Service Level (CSL)": csl,
        "Z-Score (Z)": z_score,
        "On-Hand Inventory (OH)": oh,
        "On-Order Inventory (OO)": oo,
        "Backorders (BO)": bo,
        "Inventory Position (IP)": ip,
        "Safety Stock (SS)": safety_stock,
    }
    inputs_extracted = {k: v for k, v in raw_inputs.items() if v is not None}

    variable_definitions = {
        "P": "Review period / cycle interval (time between periodic reviews)",
        "L": "Replenishment lead time",
        "P + L": "Protection period / interval (total vulnerability duration)",
        "d": "Average demand rate per unit time",
        "sigma_d": "Standard deviation of demand per unit time",
        "CSL": "Cycle service level",
        "Z": "Standard normal safety factor",
        "SS": "Safety stock buffer = Z * sigma_d * sqrt(P + L)",
        "S": "Target order-up-to inventory level = d * (P + L) + SS",
        "IP": "Current inventory position = On-Hand + On-Order - Backorders",
        "Q": "Recommended replenishment order quantity = max(0, S - IP)",
    }

    formula = (
        "Protection Period = P + L\n"
        "Expected Protection Demand = d * (P + L)\n"
        "Safety Stock (SS) = Z * sigma_d * sqrt(P + L)\n"
        "Target Inventory (S) = d * (P + L) + SS\n"
        "Inventory Position (IP) = On_Hand + On_Order - Backorders\n"
        "Order Quantity (Q) = max(0, S - IP)"
    )

    formula_substituted = (
        f"Protection Period = {review_period:g} + {lead_time:g} = {protection_period:g};\n"
        f"Expected Protection Demand = {demand_mean:g} * {protection_period:g} = {prot_mean:.2f};\n"
        f"Safety Stock (SS) = {z_score:.4f} * {sigma_d:g} * sqrt({protection_period:g}) = {ss:.2f};\n"
        f"Target Stock (S) = {prot_mean:.2f} + {ss:.2f} = {target_inventory:.2f};\n"
        f"Inventory Position (IP) = {oh:g} + {oo:g} - {bo:g} = {ip:.2f};\n"
        f"Order Quantity (Q) = max(0, {target_inventory:.2f} - {ip:.2f}) = {order_qty:.2f} (Ceiling: {rec_order})"
    )

    return PeriodicReviewResult(
        final_answer=rec_order,
        order_quantity=order_qty,
        recommended_order_quantity=rec_order,
        order_up_to_level=target_inventory,
        recommended_order_up_to_level=rec_target,
        safety_stock=ss,
        recommended_safety_stock=rec_ss,
        review_period_days=review_period,
        lead_time_days=lead_time,
        protection_period_days=protection_period,
        daily_demand_mean=demand_mean,
        daily_demand_std=sigma_d,
        protection_demand_mean=prot_mean,
        protection_demand_std=prot_std,
        service_level=csl,
        z_score=z_score,
        current_on_hand=oh,
        current_on_order=oo,
        current_backorders=bo,
        inventory_position=ip,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        verification=verification,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_substituted=formula_substituted,
        verification_checks=checks_passed,
        protection_interval_days=protection_period,
    )


# Backward-compatible aliases
calculate_fixed_period_review = calculate_periodic_review
FixedPeriodReviewResult = PeriodicReviewResult


# ---------------------------------------------------------------------------
# 5. Newsvendor Model (Single-Period Stochastic Inventory)
# ---------------------------------------------------------------------------

@dataclass
class NewsvendorResult:
    final_answer: int
    optimal_quantity: float
    recommended_quantity: int
    underage_cost: float
    overage_cost: float
    critical_ratio: float
    z_score: float
    unit_price: float
    unit_cost: float
    salvage_value: float
    shortage_penalty: float
    demand_mean: float
    demand_std: float
    expected_lost_sales: float
    expected_leftover_units: float
    expected_sales: float
    expected_revenue: float
    expected_cost: float
    expected_salvage_recovery: float
    expected_shortage_penalty: float
    expected_profit: float
    in_stock_probability: float
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_substituted: str
    verification_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_verification_details(self) -> Dict[str, Any]:
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_substituted": self.formula_substituted,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_newsvendor(
    price: Optional[float] = None,
    cost: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    salvage_value: Optional[float] = 0.0,
    shortage_penalty: Optional[float] = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    distribution: str = "normal",
) -> NewsvendorResult:
    """
    Calculates deterministic Newsvendor Model for single-period stochastic inventory:
        Cost of underage (Cu) = price - cost + shortage_penalty (or direct underage_cost)
        Cost of overage (Co)  = cost - salvage_value (or direct overage_cost)
        Critical ratio (CR)   = Cu / (Cu + Co)
        Normal distribution:
            z = NormalInvCDF(CR)
            Q* = demand_mean + z * demand_std
            
    Strict Validation:
        - Either (price and cost), OR (underage_cost and overage_cost) must be provided
        - demand_mean > 0, demand_std >= 0
        - Never invents missing numerical inputs
    """
    # 1. Input validation
    if demand_mean is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (Expected demand mean mu).")
    if demand_std is None:
        raise ValueError("Missing required numerical input: 'demand_std' (Demand standard deviation sigma).")

    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be strictly positive (> 0), got {demand_mean}.")
    if demand_std < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {demand_std}.")

    s_val = 0.0 if salvage_value is None else float(salvage_value)
    g_val = 0.0 if shortage_penalty is None else float(shortage_penalty)

    # Determine Cu and Co
    if underage_cost is not None and overage_cost is not None:
        if underage_cost <= 0:
            raise ValueError(f"Underage cost must be strictly positive (> 0), got {underage_cost}.")
        if overage_cost <= 0:
            raise ValueError(f"Overage cost must be strictly positive (> 0), got {overage_cost}.")
        cu = float(underage_cost)
        co = float(overage_cost)
        p_val = price if price is not None else (cu + co)
        c_val = cost if cost is not None else co
    else:
        if price is None:
            raise ValueError("Missing required numerical input: 'price' (Selling price p per unit, or underage_cost).")
        if cost is None:
            raise ValueError("Missing required numerical input: 'cost' (Unit cost c, or overage_cost).")
        if price <= 0:
            raise ValueError(f"Selling price must be strictly positive (> 0), got {price}.")
        if cost <= 0:
            raise ValueError(f"Unit cost must be strictly positive (> 0), got {cost}.")
        if cost >= price:
            raise ValueError(f"Unit cost (${cost:.2f}) must be strictly less than selling price (${price:.2f}) for a viable product.")
        if s_val > cost:
            raise ValueError(f"Salvage value (${s_val:.2f}) cannot exceed unit cost (${cost:.2f}).")
        if g_val < 0:
            raise ValueError(f"Shortage penalty cannot be negative, got {g_val}.")

        p_val = float(price)
        c_val = float(cost)
        cu = p_val - c_val + g_val
        co = c_val - s_val

    # 2. Intermediate calculations
    cr = cu / (cu + co)

    if distribution.lower() == "normal":
        if demand_std == 0.0:
            z_score = 0.0
            q_opt = float(demand_mean)
            exp_lost_sales = 0.0
            exp_leftover = 0.0
            exp_sales = float(demand_mean)
        else:
            z_score = normal_inv_cdf(cr)
            q_opt = max(0.0, demand_mean + z_score * demand_std)
            # Standard loss function for expected shortage
            l_z = standard_normal_loss_function(z_score)
            exp_lost_sales = demand_std * l_z
            exp_leftover = max(0.0, q_opt - demand_mean + exp_lost_sales)
            exp_sales = max(0.0, demand_mean - exp_lost_sales)
    else:
        raise NotImplementedError(f"Distribution '{distribution}' not supported. Use 'normal'.")

    rec_q = max(0, int(round(q_opt)))

    exp_rev = p_val * exp_sales
    exp_c = c_val * q_opt
    exp_salv = s_val * exp_leftover
    exp_pen = g_val * exp_lost_sales
    exp_prof = exp_rev + exp_salv - exp_c - exp_pen
    in_stock_prob = normal_cdf(z_score) if demand_std > 0 else 1.0

    intermediates = {
        "cost_of_underage (Cu = p - c + g)": cu,
        "cost_of_overage (Co = c - s)": co,
        "critical_ratio (Cu / (Cu + Co))": cr,
        "z_score (NormalInvCDF(CR))": z_score,
        "exact_optimal_quantity (Q*)": q_opt,
        "expected_sales_units": exp_sales,
        "expected_leftover_units": exp_leftover,
        "expected_lost_sales_units": exp_lost_sales,
        "expected_revenue": exp_rev,
        "expected_purchase_cost": exp_c,
        "expected_salvage_recovery": exp_salv,
        "expected_shortage_penalty": exp_pen,
        "expected_total_profit": exp_prof,
        "in_stock_probability": in_stock_prob,
    }

    assumptions = [
        "Single-period inventory decision with a single procurement opportunity before demand materializes.",
        "Unsold units cannot be carried over to future periods and are liquidated for salvage value s.",
        "Unsatisfied customer demand results in lost sales with lost contribution margin (p - c) plus penalty g.",
        "Demand is stationary and follows a normal distribution N(mu, sigma).",
        "The decision objective is to maximize total expected financial profit.",
    ]

    checks_passed = [
        f"Economic condition satisfied: Price (${p_val:.2f}) > Cost (${c_val:.2f}) >= Salvage (${s_val:.2f}).",
        f"Cost of underage Cu = ${cu:.2f}, Cost of overage Co = ${co:.2f}.",
        f"Critical ratio / optimal service level CR = {cr:.4f} ({cr*100:.2f}%).",
        f"Optimal production/order quantity Q* = {rec_q} units (Z = {z_score:.4f}).",
        f"Expected inventory flow balance: Expected Sales ({exp_sales:.2f}) + Expected Leftovers ({exp_leftover:.2f}) = Q* ({q_opt:.2f}).",
    ]
    verification = VerificationResult(
        is_valid=True,
        checks_passed=checks_passed,
    )

    raw_inputs = {
        "Selling Price (p)": price,
        "Unit Cost (c)": cost,
        "Salvage Value (s)": salvage_value,
        "Shortage Penalty (g)": shortage_penalty,
        "Cost of Underage (Cu)": cu,
        "Cost of Overage (Co)": co,
        "Demand Mean (mu)": demand_mean,
        "Demand Std Dev (sigma)": demand_std,
        "Distribution": distribution,
    }
    inputs_extracted = {k: v for k, v in raw_inputs.items() if v is not None}

    variable_definitions = {
        "p": "Selling price per unit ($/unit)",
        "c": "Unit purchase or production cost ($/unit)",
        "s": "Salvage value obtained per unsold unit ($/unit)",
        "g": "Shortage penalty for lost customer goodwill ($/unit)",
        "Cu": "Cost of underage (lost profit per unit of unmet demand = p - c + g)",
        "Co": "Cost of overage (loss per unit of excess leftover inventory = c - s)",
        "CR": "Critical ratio / critical fractile = Cu / (Cu + Co)",
        "mu": "Mean expected demand during single selling season",
        "sigma": "Standard deviation of seasonal demand",
        "Z": "Standard normal inverse CDF quantile for target fractile CR",
        "Q*": "Optimal production/order quantity = mu + Z * sigma",
        "E[Profit]": "Expected total financial profit ($)",
    }

    formula = (
        "Cost of Underage: Cu = p - c + g\n"
        "Cost of Overage: Co = c - s\n"
        "Critical Ratio: CR = Cu / (Cu + Co)\n"
        "Optimal Order Quantity: Q* = mu + NormalInvCDF(CR) * sigma\n"
        "Expected Profit: E[Profit] = p * E[Sales] + s * E[Leftovers] - c * Q* - g * E[LostSales]"
    )

    formula_substituted = (
        f"Cu = ${p_val:.2f} - ${c_val:.2f} + ${g_val:.2f} = ${cu:.2f};\n"
        f"Co = ${c_val:.2f} - ${s_val:.2f} = ${co:.2f};\n"
        f"CR = ${cu:.2f} / (${cu:.2f} + ${co:.2f}) = {cr:.4f} ({cr*100:.2f}% fractile);\n"
        f"Z = NormalInvCDF({cr:.4f}) = {z_score:.4f};\n"
        f"Q* = {demand_mean:g} + ({z_score:.4f} * {demand_std:g}) = {q_opt:.2f} (Rounded: {rec_q});\n"
        f"Expected Profit = ${exp_prof:,.2f}"
    )

    return NewsvendorResult(
        final_answer=rec_q,
        optimal_quantity=q_opt,
        recommended_quantity=rec_q,
        underage_cost=cu,
        overage_cost=co,
        critical_ratio=cr,
        z_score=z_score,
        unit_price=p_val,
        unit_cost=c_val,
        salvage_value=s_val,
        shortage_penalty=g_val,
        demand_mean=demand_mean,
        demand_std=demand_std,
        expected_lost_sales=exp_lost_sales,
        expected_leftover_units=exp_leftover,
        expected_sales=exp_sales,
        expected_revenue=exp_rev,
        expected_cost=exp_c,
        expected_salvage_recovery=exp_salv,
        expected_shortage_penalty=exp_pen,
        expected_profit=exp_prof,
        in_stock_probability=in_stock_prob,
        intermediate_calculations=intermediates,
        assumptions=assumptions,
        verification=verification,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_substituted=formula_substituted,
        verification_checks=checks_passed,
    )
