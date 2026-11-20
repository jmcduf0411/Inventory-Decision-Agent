"""
Comprehensive Unit Test Suite for Deterministic Inventory Decision Tools
========================================================================
Validates:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (All 4 variability cases)
4. calculate_periodic_review()
5. calculate_newsvendor()

Ensures each tool:
- Validates inputs strictly (raises ValueError for invalid or missing inputs)
- Returns structured results with final_answer
- Returns complete intermediate_calculations
- Identifies underlying operations research assumptions
- Never invents missing numerical inputs
- Integrates with the end-to-end Inventory Decision Agent
"""

import math
import os
import sys
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import inventory_tools as tools
from inventory_agent import InventoryDecisionAgent


# ===========================================================================
# 1. calculate_eoq() Tests
# ===========================================================================

class TestCalculateEOQ:
    def test_eoq_textbook_benchmark(self):
        """D=10,000, S=$50, H=$2 -> Q*=707.11 -> final_answer=708."""
        res = tools.calculate_eoq(demand=10000, order_cost=50, holding_cost=2)

        assert res.final_answer == 708
        assert res.recommended_order_quantity == 708
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 707.106
        assert pytest.approx(res.annual_ordering_cost, rel=1e-2) == 707.11
        assert pytest.approx(res.annual_holding_cost, rel=1e-2) == 707.11
        assert pytest.approx(res.annual_inventory_cost, rel=1e-2) == 1414.21
        assert res.cost_difference_holding_vs_ordering < 0.01

        # Check required fields
        assert isinstance(res.intermediate_calculations, dict)
        assert "exact_optimal_q" in res.intermediate_calculations
        assert "annual_ordering_cost" in res.intermediate_calculations
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) > 0
        assert any("demand is deterministic" in a.lower() for a in res.assumptions)

    def test_eoq_with_percentage_holding_cost(self):
        """H derived from holding_rate=20% and unit_cost=$25 -> H=$5.00."""
        res = tools.calculate_eoq(demand=12000, order_cost=100, unit_cost=25, holding_rate=20.0)

        assert res.holding_cost_per_unit == 5.0
        assert res.final_answer == 693
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 692.82
        assert res.annual_purchase_cost == 12000 * 25
        assert "holding_cost_used" in res.intermediate_calculations

    def test_eoq_missing_numerical_inputs_raises_error(self):
        """Never invent missing numerical inputs: missing demand, order_cost, or holding_cost."""
        with pytest.raises(ValueError, match="Missing required numerical input: 'demand'"):
            tools.calculate_eoq(demand=None, order_cost=50, holding_cost=2)

        with pytest.raises(ValueError, match="Missing required numerical input: 'order_cost'"):
            tools.calculate_eoq(demand=1000, order_cost=None, holding_cost=2)

        with pytest.raises(ValueError, match="Missing required holding cost input"):
            tools.calculate_eoq(demand=1000, order_cost=50, holding_cost=None)

    def test_eoq_invalid_numbers_raise_error(self):
        with pytest.raises(ValueError, match="Annual demand must be strictly positive"):
            tools.calculate_eoq(demand=-500, order_cost=50, holding_cost=2)

        with pytest.raises(ValueError, match="Order cost must be strictly positive"):
            tools.calculate_eoq(demand=1000, order_cost=0, holding_cost=2)

        with pytest.raises(ValueError, match="Holding cost per unit must be strictly positive"):
            tools.calculate_eoq(demand=1000, order_cost=50, holding_cost=-1)


# ===========================================================================
# 2. calculate_quantity_discount() Tests
# ===========================================================================

class TestCalculateQuantityDiscount:
    @pytest.fixture
    def three_tiers(self):
        return [
            (1.0, 999.0, 5.00),
            (1000.0, 1999.0, 4.80),
            (2000.0, None, 4.75),
        ]

    def test_quantity_discount_percentage_holding_cost(self, three_tiers):
        """
        D = 5000, S = 49, holding_rate = 20%
        Tier 1: Q* unconstrained = 700 (feasible, Cost = $25,700)
        Tier 2: Q* unconstrained = 714.43 (infeasible -> adjust to 1000, Cost = $24,725)
        Tier 3: Q* unconstrained = 718.18 (infeasible -> adjust to 2000, Cost = $24,822.50)
        Optimal: Tier 2, Q = 1000, Total Cost = $24,725.
        """
        res = tools.calculate_quantity_discount(
            demand=5000,
            order_cost=49,
            discount_tiers=three_tiers,
            holding_rate=20.0
        )

        assert res.final_answer == 1000
        assert res.recommended_order_quantity == 1000
        assert res.best_tier.tier_number == 2
        assert res.best_tier.unit_price == 4.80
        assert pytest.approx(res.total_annual_cost, rel=1e-2) == 24725.00
        assert pytest.approx(res.cost_savings_vs_base, rel=1e-2) == 975.00

        # Structured result fields
        assert isinstance(res.intermediate_calculations, dict)
        assert "tier_step_by_step" in res.intermediate_calculations
        assert len(res.intermediate_calculations["tier_step_by_step"]) == 3
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) > 0

    def test_quantity_discount_fixed_holding_cost(self, three_tiers):
        """Fixed holding cost of $1.00/unit across all tiers."""
        res = tools.calculate_quantity_discount(
            demand=5000,
            order_cost=49,
            discount_tiers=three_tiers,
            holding_cost=1.00
        )

        assert res.final_answer > 0
        assert "Fixed holding cost" in res.holding_cost_mode
        assert res.verification.is_valid is True

    def test_quantity_discount_missing_or_invalid_inputs(self, three_tiers):
        with pytest.raises(ValueError, match="Missing required numerical input: 'demand'"):
            tools.calculate_quantity_discount(demand=None, order_cost=50, discount_tiers=three_tiers, holding_cost=1)

        with pytest.raises(ValueError, match="Missing required input: 'discount_tiers'"):
            tools.calculate_quantity_discount(demand=1000, order_cost=50, discount_tiers=[], holding_cost=1)

        with pytest.raises(ValueError, match="Must provide either 'holding_cost'"):
            tools.calculate_quantity_discount(demand=1000, order_cost=50, discount_tiers=three_tiers)


# ===========================================================================
# 3. calculate_rop() Tests (All 4 Cases)
# ===========================================================================

class TestCalculateROP:
    def test_case_1_constant_demand_constant_lead_time(self):
        """Case 1: sigma_d = 0, sigma_L = 0 -> sigma_LT = 0, SS = 0, ROP = d * L."""
        res = tools.calculate_rop(demand_mean=50, lead_time=9, demand_std=0.0, lead_time_std=0.0)

        assert "Case 1" in res.case_description
        assert res.lead_time_demand_std == 0.0
        assert res.safety_stock == 0.0
        assert res.recommended_safety_stock == 0
        assert res.reorder_point == 450.0
        assert res.final_answer == 450
        assert res.intermediate_calculations["total_lead_time_std_dev (sigma_LT)"] == 0.0
        assert len(res.assumptions) > 0

    def test_case_2_variable_demand_constant_lead_time(self):
        """
        Case 2: sigma_d = 10, sigma_L = 0, d = 50, L = 9, CSL = 95%.
        sigma_LT = 10 * sqrt(9) = 30
        SS = 1.6449 * 30 = 49.35 -> 50
        ROP = 450 + 49.35 = 499.35 -> 500
        """
        res = tools.calculate_rop(
            demand_mean=50, lead_time=9, demand_std=10.0, lead_time_std=0.0, service_level=0.95
        )

        assert "Case 2" in res.case_description
        assert pytest.approx(res.lead_time_demand_std, rel=1e-3) == 30.0
        assert pytest.approx(res.safety_stock, rel=1e-2) == 49.35
        assert res.recommended_safety_stock == 50
        assert pytest.approx(res.reorder_point, rel=1e-2) == 499.35
        assert res.final_answer == 500
        assert res.demand_variance_component == 9 * 100
        assert res.lead_time_variance_component == 0.0

    def test_case_3_constant_demand_variable_lead_time(self):
        """
        Case 3: sigma_d = 0, sigma_L = 2, d = 50, L = 9, CSL = 95%.
        sigma_LT = d * sigma_L = 50 * 2 = 100
        SS = 1.6449 * 100 = 164.49 -> 165
        ROP = 450 + 164.49 = 614.49 -> 615
        """
        res = tools.calculate_rop(
            demand_mean=50, lead_time=9, demand_std=0.0, lead_time_std=2.0, service_level=0.95
        )

        assert "Case 3" in res.case_description
        assert pytest.approx(res.lead_time_demand_std, rel=1e-3) == 100.0
        assert pytest.approx(res.safety_stock, rel=1e-2) == 164.49
        assert res.recommended_safety_stock == 165
        assert pytest.approx(res.reorder_point, rel=1e-2) == 614.49
        assert res.final_answer == 615
        assert res.demand_variance_component == 0.0
        assert res.lead_time_variance_component == (50 ** 2) * (2 ** 2)

    def test_case_4_variable_demand_variable_lead_time(self):
        """
        Case 4: sigma_d = 10, sigma_L = 2, d = 50, L = 9, CSL = 95%.
        var_demand = 9 * 100 = 900
        var_lt = 2500 * 4 = 10000
        sigma_LT = sqrt(10900) = 104.403
        SS = 1.6449 * 104.403 = 171.73 -> 172
        ROP = 450 + 171.73 = 621.73 -> 622
        """
        res = tools.calculate_rop(
            demand_mean=50, lead_time=9, demand_std=10.0, lead_time_std=2.0, service_level=0.95
        )

        assert "Case 4" in res.case_description
        assert pytest.approx(res.lead_time_demand_std, rel=1e-3) == 104.403
        assert pytest.approx(res.safety_stock, rel=1e-2) == 171.73
        assert res.recommended_safety_stock == 172
        assert pytest.approx(res.reorder_point, rel=1e-2) == 621.73
        assert res.final_answer == 622

    def test_rop_with_explicit_z_input(self):
        """Testing direct z-score pass (e.g. z=2.0)."""
        res = tools.calculate_rop(demand_mean=50, lead_time=4, demand_std=5.0, z=2.0)
        # sigma_LT = 5 * sqrt(4) = 10. SS = 2 * 10 = 20. ROP = 200 + 20 = 220.
        assert res.z_score == 2.0
        assert res.safety_stock == 20.0
        assert res.reorder_point == 220.0
        assert res.final_answer == 220

    def test_rop_missing_inputs_raises_error(self):
        with pytest.raises(ValueError, match="Missing required numerical input: 'demand_mean'"):
            tools.calculate_rop(demand_mean=None, lead_time=9)

        with pytest.raises(ValueError, match="Missing required numerical input: 'lead_time'"):
            tools.calculate_rop(demand_mean=50, lead_time=None)

        with pytest.raises(ValueError, match="Variability detected"):
            # Has demand variability but neither service_level nor z provided
            tools.calculate_rop(demand_mean=50, lead_time=9, demand_std=10.0)


# ===========================================================================
# 4. calculate_periodic_review() Tests
# ===========================================================================

class TestCalculatePeriodicReview:
    def test_periodic_review_order_up_to(self):
        """
        P = 14, L = 7 -> Protection period = 21.
        d = 20, sigma_d = 4, CSL = 95% (Z = 1.6449).
        Demand during protection = 20 * 21 = 420.
        sigma_protection = 4 * sqrt(21) = 18.3303.
        SS = 1.6449 * 18.3303 = 30.15 -> 31.
        Target inventory = 420 + 30.15 = 450.15 -> 451.
        On-hand = 120, on-order = 50, backorders = 10 -> IP = 160.
        Order quantity = max(0, 450.15 - 160) = 290.15 -> 291.
        """
        res = tools.calculate_periodic_review(
            review_period=14,
            lead_time=7,
            demand_mean=20,
            demand_std=4,
            service_level=0.95,
            on_hand=120,
            on_order=50,
            backorders=10
        )

        assert res.protection_period_days == 21.0
        assert pytest.approx(res.protection_demand_mean, rel=1e-3) == 420.0
        assert pytest.approx(res.safety_stock, rel=1e-2) == 30.15
        assert res.recommended_safety_stock == 31
        assert pytest.approx(res.order_up_to_level, rel=1e-2) == 450.15
        assert res.recommended_order_up_to_level == 451
        assert res.inventory_position == 160.0
        assert pytest.approx(res.order_quantity, rel=1e-2) == 290.15
        assert res.final_answer == 291

        # Intermediate calculations and assumptions
        assert "target_inventory_level (S)" in res.intermediate_calculations
        assert "constrained_order_quantity max(0, S - IP)" in res.intermediate_calculations
        assert len(res.assumptions) > 0

    def test_periodic_review_excess_inventory_yields_zero_order(self):
        """If inventory position exceeds target inventory, order quantity is capped at 0."""
        res = tools.calculate_periodic_review(
            review_period=7,
            lead_time=3,
            demand_mean=10,
            demand_std=2,
            service_level=0.95,
            on_hand=500,
            on_order=0,
            backorders=0
        )
        assert res.order_quantity == 0.0
        assert res.final_answer == 0

    def test_periodic_review_missing_inputs_raises_error(self):
        with pytest.raises(ValueError, match="Missing required numerical input: 'review_period'"):
            tools.calculate_periodic_review(review_period=None, lead_time=7, demand_mean=20)

        with pytest.raises(ValueError, match="Missing required numerical input: 'lead_time'"):
            tools.calculate_periodic_review(review_period=14, lead_time=None, demand_mean=20)

        with pytest.raises(ValueError, match="Missing required numerical input: 'demand_mean'"):
            tools.calculate_periodic_review(review_period=14, lead_time=7, demand_mean=None)


# ===========================================================================
# 5. calculate_newsvendor() Tests
# ===========================================================================

class TestCalculateNewsvendor:
    def test_newsvendor_benchmark(self):
        """
        Price p = 100, Cost c = 40, Salvage s = 20, Shortage penalty g = 0.
        Cu = 100 - 40 = 60
        Co = 40 - 20 = 20
        CR = 60 / (60 + 20) = 0.75
        Demand: Normal(mu=200, sigma=40)
        Z = inv_cdf(0.75) = 0.67449
        Q* = 200 + 0.67449 * 40 = 226.98 -> 227
        """
        res = tools.calculate_newsvendor(
            price=100,
            cost=40,
            salvage_value=20,
            shortage_penalty=0,
            demand_mean=200,
            demand_std=40,
            distribution="normal"
        )

        assert res.underage_cost == 60.0
        assert res.overage_cost == 20.0
        assert pytest.approx(res.critical_ratio, rel=1e-3) == 0.75
        assert pytest.approx(res.z_score, rel=1e-3) == 0.6745
        assert pytest.approx(res.optimal_quantity, rel=1e-2) == 226.98
        assert res.final_answer == 227
        assert res.expected_profit > 0
        assert res.expected_lost_sales > 0
        assert res.expected_leftover_units > 0

        # Intermediates and assumptions
        assert "cost_of_underage (Cu = p - c + g)" in res.intermediate_calculations
        assert "critical_ratio (Cu / (Cu + Co))" in res.intermediate_calculations
        assert len(res.assumptions) > 0

    def test_newsvendor_with_shortage_penalty(self):
        """p=50, c=20, s=5, g=10 -> Cu = 50 - 20 + 10 = 40, Co = 20 - 5 = 15 -> CR = 40/55 = 0.7273."""
        res = tools.calculate_newsvendor(
            price=50, cost=20, salvage_value=5, shortage_penalty=10, demand_mean=100, demand_std=15
        )
        assert res.underage_cost == 40.0
        assert res.overage_cost == 15.0
        assert pytest.approx(res.critical_ratio, rel=1e-3) == 40.0 / 55.0
        assert res.final_answer > 100

    def test_newsvendor_missing_or_unprofitable_inputs_raises_error(self):
        with pytest.raises(ValueError, match="Missing required numerical input: 'price'"):
            tools.calculate_newsvendor(price=None, cost=40, demand_mean=200, demand_std=40)

        with pytest.raises(ValueError, match="Unit cost .* must be strictly less than selling price"):
            tools.calculate_newsvendor(price=30, cost=50, demand_mean=100, demand_std=10)

        with pytest.raises(ValueError, match="Salvage value .* cannot exceed unit cost"):
            tools.calculate_newsvendor(price=100, cost=40, salvage_value=50, demand_mean=100, demand_std=10)


# ===========================================================================
# 6. End-to-End Agent Integration Tests
# ===========================================================================

class TestInventoryDecisionAgentIntegration:
    @pytest.fixture
    def agent(self):
        return InventoryDecisionAgent()

    def test_agent_eoq(self, agent):
        q = "Annual demand is 12,000 units, order cost is $100, and holding cost is $4. What is our EOQ?"
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "eoq"
        assert resp.raw_result.final_answer == 775

    def test_eoq_holding_cost_not_extracted_as_unit_purchase_cost_regression(self, agent):
        """
        Regression Test:
        'Our annual demand is 12,000 units. Ordering cost is $100 per order, and holding cost is $4 per unit per year.
         What is our optimal order quantity and how often should we place orders?'

        Holding cost ($4 per unit per year) must NOT be extracted as Unit Purchase Cost (C).
        Since purchase price is unspecified:
        - 'unit_cost' must not be in extracted_inputs
        - raw_result.unit_purchase_cost must be None
        - raw_result.annual_purchase_cost must be None
        - total_annual_cost must equal annual_inventory_cost (holding + ordering)
        - EOQ = 774.60 exact -> final_answer = 775
        - All 8 verification components must be present
        """
        q = (
            "Our annual demand is 12,000 units. Ordering cost is $100 per order, "
            "and holding cost is $4 per unit per year. "
            "What is our optimal order quantity and how often should we place orders?"
        )
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "eoq"
        assert resp.extracted_inputs["demand"] == 12000.0
        assert resp.extracted_inputs["order_cost"] == 100.0
        assert resp.extracted_inputs["holding_cost"] == 4.0
        assert "unit_cost" not in resp.extracted_inputs, "Holding cost was incorrectly extracted as unit_cost!"

        res = resp.raw_result
        assert res.unit_purchase_cost is None
        assert res.annual_purchase_cost is None
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 774.597
        assert res.final_answer == 775
        assert res.recommended_order_quantity == 775
        assert pytest.approx(res.annual_ordering_cost, rel=1e-2) == 1549.19
        assert pytest.approx(res.annual_holding_cost, rel=1e-2) == 1549.19
        assert pytest.approx(res.annual_inventory_cost, rel=1e-2) == 3098.39
        assert res.total_annual_cost == res.annual_inventory_cost

        # Verify all 8 components
        assert "Unit Purchase Cost (C)" not in res.inputs_extracted
        assert res.inputs_extracted["Annual Demand (D)"] == 12000.0
        assert res.inputs_extracted["Holding Cost per Unit (H)"] == 4.0
        assert "Q*" in res.variable_definitions
        assert "sqrt((2 * D * S) / H)" in res.formula
        assert "12000" in res.formula_substituted
        assert "exact_optimal_q" in res.intermediate_calculations
        assert res.final_answer == 775
        assert len(res.verification_checks) >= 3
        assert len(res.assumptions) >= 5

    def test_agent_quantity_discount(self, agent):
        q = "Annual demand 5000, order cost $49, holding rate 20%. Tiers: 1-999: $5.00, 1000-1999: $4.80, 2000+: $4.75."
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "quantity_discount"
        assert resp.raw_result.final_answer == 1000

    def test_agent_rop(self, agent):
        q = "Daily demand is 200 units with std dev 25. Supplier lead time is 9 days. What is our ROP for 95% service level?"
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "safety_stock_rop"
        assert resp.raw_result.final_answer > 0

    def test_agent_periodic_review(self, agent):
        q = "Review inventory every 14 days, lead time 7 days, daily demand 50, std dev 8, 98% service level. On-hand 450, on-order 150, backorders 20."
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "fixed_period_review"
        assert resp.raw_result.final_answer > 0

    def test_agent_newsvendor(self, agent):
        q = "We sell holiday gift baskets for $75 that cost $35 to make. Leftovers can be salvaged for $15. Demand mean is 500 with standard deviation of 80."
        resp = agent.ask(q)
        assert resp.success is True
        assert resp.identified_model == "newsvendor"
        assert resp.raw_result.final_answer > 0

    @pytest.mark.parametrize("phrase,expected_order_cost", [
        ("The ordering cost is $50", 50.0),
        ("With an order cost of $60", 60.0),
        ("A setup cost of $45 per run", 45.0),
        ("Our setup expense is $70", 70.0),
        ("Incurring a cost per order of $55", 55.0),
        ("The cost of placing an order is $65", 65.0),
        ("Our ordering/setup cost is $80", 80.0),
        ("Our order/setup cost is $75", 75.0),
        ("Placing an order costs $40", 40.0),
    ])
    def test_eoq_ordering_cost_synonyms(self, agent, phrase, expected_order_cost):
        """Validates all required synonyms for EOQ ordering cost."""
        q = f"Annual demand is 10,000 units. {phrase}. Holding cost is $2.00 per unit per year."
        resp = agent.ask(q)
        assert resp.success is True, f"Failed for phrase: '{phrase}'. Missing: {resp.missing_inputs}"
        assert resp.extracted_inputs["order_cost"] == expected_order_cost
        assert resp.raw_result.final_answer > 0

    def test_distinguish_rop_parameters(self, agent):
        """Distinguishes average demand vs demand std dev, and average lead time vs lead-time std dev."""
        q = (
            "Our average demand is 100 units/day with a demand standard deviation of 15 units. "
            "The average lead time is 10 days with a lead-time standard deviation of 2 days. "
            "Target cycle service level is 95%."
        )
        resp = agent.ask(q)
        assert resp.success is True, f"Failed ROP distinction: {resp.missing_inputs}"
        assert resp.extracted_inputs["demand_mean"] == 100.0
        assert resp.extracted_inputs["demand_std"] == 15.0
        assert resp.extracted_inputs["lead_time"] == 10.0
        assert resp.extracted_inputs["lead_time_std"] == 2.0
        assert resp.extracted_inputs["service_level"] == 0.95
        assert resp.raw_result.has_variable_lead_time is True
        assert "Case 4" in resp.raw_result.case_description

    def test_distinguish_quantity_discount_parameters(self, agent):
        """Distinguishes quantity breaks, unit prices, ordering cost, fixed vs percentage holding cost."""
        q = (
            "Annual demand of 8,000 units with cost per order of $60. "
            "Percentage holding cost is 20% on unit purchase price. "
            "Quantity breaks: 1-499: $10.00, 500-999: $9.50, 1000+: $9.00."
        )
        resp = agent.ask(q)
        assert resp.success is True, f"Failed QD distinction: {resp.missing_inputs}"
        assert resp.extracted_inputs["demand"] == 8000.0
        assert resp.extracted_inputs["order_cost"] == 60.0
        assert resp.extracted_inputs["holding_rate"] == 0.20
        assert len(resp.extracted_inputs["discount_tiers"]) == 3
        assert resp.raw_result.final_answer > 0

    def test_distinguish_fixed_period_review_parameters(self, agent):
        """Distinguishes review period, lead time, on-hand, on-order, and backorders."""
        q = (
            "Review period is 14 days, supplier lead time is 7 days. "
            "Average daily demand is 25 units with a standard deviation of 5 units. "
            "On-hand inventory is 150 units, on-order inventory is 80 units, and backorders are 20 units. "
            "Desired service level is 95%."
        )
        resp = agent.ask(q)
        assert resp.success is True, f"Failed FPR distinction: {resp.missing_inputs}"
        assert resp.extracted_inputs["review_period"] == 14.0
        assert resp.extracted_inputs["lead_time"] == 7.0
        assert resp.extracted_inputs["demand_mean"] == 25.0
        assert resp.extracted_inputs["demand_std"] == 5.0
        assert resp.extracted_inputs["on_hand"] == 150.0
        assert resp.extracted_inputs["on_order"] == 80.0
        assert resp.extracted_inputs["backorders"] == 20.0
        # IP = 150 + 80 - 20 = 210
        assert resp.raw_result.inventory_position == 210.0
        assert resp.raw_result.final_answer > 0

    def test_distinguish_newsvendor_parameters(self, agent):
        """Distinguishes selling price, purchase cost, salvage value, mean demand, demand std dev."""
        q = (
            "Selling price is $120 per unit. Unit purchase cost is $50. "
            "Salvage value of unsold items is $20. "
            "Mean demand is 400 with a demand standard deviation of 60 units. "
            "Demand distribution is normal."
        )
        resp = agent.ask(q)
        assert resp.success is True, f"Failed Newsvendor distinction: {resp.missing_inputs}"
        assert resp.extracted_inputs["price"] == 120.0
        assert resp.extracted_inputs["cost"] == 50.0
        assert resp.extracted_inputs["salvage_value"] == 20.0
        assert resp.extracted_inputs["demand_mean"] == 400.0
        assert resp.extracted_inputs["demand_std"] == 60.0
        assert resp.extracted_inputs["distribution"] == "normal"
        assert resp.raw_result.underage_cost == 120.0 - 50.0  # Cu = 70
        assert resp.raw_result.overage_cost == 50.0 - 20.0   # Co = 30
        assert pytest.approx(resp.raw_result.critical_ratio, rel=1e-3) == 70.0 / 100.0  # 0.70
        assert resp.raw_result.final_answer > 400


# ===========================================================================
# 6. Verifiable Answer Architecture Tests (Prompt 4 Requirements)
# ===========================================================================

class TestVerifiableCalculationOutputs:
    """
    Tests that every calculation tool returns all 8 required verification components:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """

    def test_eoq_returns_all_8_verifiable_components(self):
        res = tools.calculate_eoq(demand=12000, order_cost=100, holding_cost=4.0)

        # 1. Inputs Extracted
        assert isinstance(res.inputs_extracted, dict)
        assert res.inputs_extracted["Annual Demand (D)"] == 12000
        assert res.inputs_extracted["Order/Setup Cost (S)"] == 100
        assert res.inputs_extracted["Holding Cost per Unit (H)"] == 4.0

        # 2. Variable Definitions
        assert isinstance(res.variable_definitions, dict)
        assert "D" in res.variable_definitions
        assert "S" in res.variable_definitions
        assert "H" in res.variable_definitions
        assert "Q*" in res.variable_definitions

        # 3. Formula
        assert isinstance(res.formula, str)
        assert "sqrt((2 * D * S) / H)" in res.formula

        # 4. Formula with Substituted Values
        assert isinstance(res.formula_substituted, str)
        assert "12000" in res.formula_substituted
        assert "100" in res.formula_substituted
        assert "4" in res.formula_substituted

        # 5. Intermediate Calculations
        assert isinstance(res.intermediate_calculations, dict)
        assert "exact_optimal_q" in res.intermediate_calculations
        assert "annual_ordering_cost" in res.intermediate_calculations
        assert "annual_holding_cost" in res.intermediate_calculations
        assert "annual_inventory_cost (Hold + Order)" in res.intermediate_calculations

        # 6. Final Answer
        assert res.final_answer == 775
        assert res.recommended_order_quantity == 775

        # 7. Verification Checks
        assert isinstance(res.verification_checks, list)
        assert len(res.verification_checks) >= 3
        assert any("Strict positivity" in chk for chk in res.verification_checks)
        assert any("Economic balance" in chk for chk in res.verification_checks)

        # 8. Assumptions
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 5
        assert any("deterministic" in a.lower() for a in res.assumptions)

        # Helper method get_verification_details()
        details = res.get_verification_details()
        assert "inputs_extracted" in details
        assert "variable_definitions" in details
        assert "formula" in details
        assert "formula_substituted" in details
        assert "intermediate_calculations" in details
        assert "final_answer" in details
        assert "verification_checks" in details
        assert "assumptions" in details

    def test_quantity_discount_returns_all_8_verifiable_components(self):
        tiers = [(1.0, 999.0, 5.00), (1000.0, 1999.0, 4.80), (2000.0, None, 4.75)]
        res = tools.calculate_quantity_discount(
            demand=5000, order_cost=49, discount_tiers=tiers, holding_rate=20.0
        )

        assert isinstance(res.inputs_extracted, dict)
        assert res.inputs_extracted["Annual Demand (D)"] == 5000
        assert res.inputs_extracted["Order/Setup Cost (S)"] == 49

        assert isinstance(res.variable_definitions, dict)
        assert "C_j" in res.variable_definitions or "D" in res.variable_definitions

        assert isinstance(res.formula, str)
        assert "sqrt((2 * D * S) / H_j)" in res.formula or "Q_j*" in res.formula

        assert isinstance(res.formula_substituted, str)
        assert "5000" in res.formula_substituted
        assert "49" in res.formula_substituted

        assert isinstance(res.intermediate_calculations, dict)
        assert "tier_step_by_step" in res.intermediate_calculations
        assert "selected_optimal_tier" in res.intermediate_calculations

        assert res.final_answer == 1000
        assert isinstance(res.verification_checks, list)
        assert len(res.verification_checks) >= 3
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 3

    def test_rop_returns_all_8_verifiable_components(self):
        res = tools.calculate_rop(
            demand_mean=50, demand_std=10, lead_time=9, lead_time_std=2, service_level=0.95
        )

        assert isinstance(res.inputs_extracted, dict)
        assert res.inputs_extracted["Average Demand (d)"] == 50
        assert res.inputs_extracted["Supplier Lead Time (L)"] == 9
        assert res.inputs_extracted["Demand Std Dev (sigma_d)"] == 10
        assert res.inputs_extracted["Lead Time Std Dev (sigma_L)"] == 2

        assert isinstance(res.variable_definitions, dict)
        assert "ROP" in res.variable_definitions
        assert "SS" in res.variable_definitions
        assert "sigma_LT" in res.variable_definitions

        assert isinstance(res.formula, str)
        assert "ROP = d * L + SS" in res.formula

        assert isinstance(res.formula_substituted, str)
        assert "50" in res.formula_substituted
        assert "9" in res.formula_substituted

        assert isinstance(res.intermediate_calculations, dict)
        assert "total_lead_time_std_dev (sigma_LT)" in res.intermediate_calculations
        assert "exact_safety_stock (SS)" in res.intermediate_calculations
        assert "exact_reorder_point (ROP)" in res.intermediate_calculations

        assert res.final_answer > 0
        assert res.final_answer == res.recommended_reorder_point

        assert isinstance(res.verification_checks, list)
        assert len(res.verification_checks) >= 3

        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 3

    def test_periodic_review_returns_all_8_verifiable_components(self):
        res = tools.calculate_periodic_review(
            review_period=14, lead_time=7, demand_mean=20, demand_std=4, service_level=0.95,
            on_hand=120, on_order=50, backorders=10
        )

        assert isinstance(res.inputs_extracted, dict)
        assert res.inputs_extracted["Review Period (P)"] == 14
        assert res.inputs_extracted["Lead Time (L)"] == 7
        assert res.inputs_extracted["Demand Mean (d)"] == 20
        assert res.inputs_extracted["On-Hand Inventory (OH)"] == 120

        assert isinstance(res.variable_definitions, dict)
        assert "P" in res.variable_definitions
        assert "S" in res.variable_definitions
        assert "IP" in res.variable_definitions
        assert "Q" in res.variable_definitions

        assert isinstance(res.formula, str)
        assert "Protection Period = P + L" in res.formula
        assert "Order Quantity (Q) = max(0, S - IP)" in res.formula

        assert isinstance(res.formula_substituted, str)
        assert "14" in res.formula_substituted
        assert "7" in res.formula_substituted
        assert "160" in res.formula_substituted  # IP = 120 + 50 - 10 = 160

        assert isinstance(res.intermediate_calculations, dict)
        assert "protection_period (P + L)" in res.intermediate_calculations
        assert "target_inventory_level (S)" in res.intermediate_calculations
        assert "current_inventory_position (IP = OH + OO - BO)" in res.intermediate_calculations

        assert res.final_answer == res.recommended_order_quantity
        assert isinstance(res.verification_checks, list)
        assert len(res.verification_checks) >= 3
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 3

    def test_newsvendor_returns_all_8_verifiable_components(self):
        res = tools.calculate_newsvendor(
            price=100, cost=40, salvage_value=20, shortage_penalty=5, demand_mean=200, demand_std=40
        )

        assert isinstance(res.inputs_extracted, dict)
        assert res.inputs_extracted["Selling Price (p)"] == 100
        assert res.inputs_extracted["Unit Cost (c)"] == 40
        assert res.inputs_extracted["Salvage Value (s)"] == 20
        assert res.inputs_extracted["Shortage Penalty (g)"] == 5

        assert isinstance(res.variable_definitions, dict)
        assert "Cu" in res.variable_definitions
        assert "Co" in res.variable_definitions
        assert "CR" in res.variable_definitions
        assert "Q*" in res.variable_definitions

        assert isinstance(res.formula, str)
        assert "CR = Cu / (Cu + Co)" in res.formula
        assert "Q* = mu + NormalInvCDF(CR) * sigma" in res.formula

        assert isinstance(res.formula_substituted, str)
        assert "100" in res.formula_substituted
        assert "40" in res.formula_substituted
        assert "200" in res.formula_substituted

        assert isinstance(res.intermediate_calculations, dict)
        assert "cost_of_underage (Cu = p - c + g)" in res.intermediate_calculations
        assert "cost_of_overage (Co = c - s)" in res.intermediate_calculations
        assert "critical_ratio (Cu / (Cu + Co))" in res.intermediate_calculations
        assert "expected_total_profit" in res.intermediate_calculations

        assert res.final_answer == res.recommended_quantity
        assert isinstance(res.verification_checks, list)
        assert len(res.verification_checks) >= 3
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 3

    def test_agent_explanation_embeds_all_8_verifiable_sections(self):
        agent = InventoryDecisionAgent()
        resp = agent.ask("Demand is 10000, ordering cost is $50, holding cost is $2. What is EOQ?")
        assert resp.success is True
        expl = resp.explanation
        assert "1. Inputs Extracted" in expl
        assert "2. Variable Definitions" in expl
        assert "3. Formula" in expl
        assert "4. Formula with Substituted Values" in expl
        assert "5. Intermediate Calculations" in expl
        assert "6. Final Answer" in expl
        assert "7. Verification Checks" in expl
        assert "8. Assumptions" in expl


# ===========================================================================
# 7. Demand Unit Conversion & BAN 517 Test 6 Regression Tests
# ===========================================================================

class TestDemandUnitConversionRegression:
    """
    Automated regression tests verifying that sub-annual demand units
    (day, week, month, quarter) are converted to annual demand
    consistently with the annual holding-cost period before EOQ calculations.
    """

    def test_professor_report_generator_test6_monthly_demand_to_annual(self):
        """
        Exact BAN 517 Project Report Generator Test 6 prompt:
        'Find EOQ. Demand is 1,000 units per month, ordering cost is $25 per order, and holding cost is $0.50 per unit per year.'
        
        Expected:
            - Monthly demand: 1,000 units/month
            - Annual demand D: 1,000 * 12 = 12,000 units/year
            - Order cost S: $25.00
            - Holding cost H: $0.50 per unit/year
            - Optimal EOQ Q* = sqrt(2 * 12,000 * 25 / 0.50) = sqrt(1,200,000) approx 1,095.45 units
            - Recommended integer Q: 1,095 or 1,096 units
            - Annual orders = 12,000 / 1095.45 approx 10.95 orders/year
            - Annual inventory cost = 2 * (1095.45 / 2 * 0.50) = $547.72
        """
        agent = InventoryDecisionAgent()
        prompt = "Find EOQ. Demand is 1,000 units per month, ordering cost is $25 per order, and holding cost is $0.50 per unit per year."
        resp = agent.ask(prompt)

        assert resp.success is True, f"Agent failed on Test 6 prompt: {resp.missing_inputs}"
        assert resp.identified_model == "eoq"

        # Check extracted inputs
        inputs = resp.extracted_inputs
        assert inputs["demand"] == 12000.0, f"Expected annual demand 12,000, got {inputs['demand']}"
        assert inputs["stated_demand"] == 1000.0, f"Expected stated demand 1,000, got {inputs.get('stated_demand')}"
        assert inputs["demand_period"] == "month", f"Expected period 'month', got {inputs.get('demand_period')}"
        assert inputs["order_cost"] == 25.0
        assert inputs["holding_cost"] == 0.50

        # Check calculation result
        res: tools.EOQResult = resp.raw_result
        assert res.annual_demand == 12000.0
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 1095.445
        assert res.recommended_order_quantity in [1095, 1096]
        assert pytest.approx(res.orders_per_year, rel=1e-2) == 10.95
        assert pytest.approx(res.annual_inventory_cost, rel=1e-2) == 547.72
        assert pytest.approx(res.annual_ordering_cost, rel=1e-2) == 273.86
        assert pytest.approx(res.annual_holding_cost, rel=1e-2) == 273.86

        # Check formula and substitutions use annual demand 12,000
        assert "12000" in res.formula_substituted
        assert "25" in res.formula_substituted
        assert "0.5" in res.formula_substituted

        # Check verifiable step components include stated demand and conversion note
        assert res.intermediate_calculations.get("stated_demand") == 1000.0
        assert res.intermediate_calculations.get("demand_period") == "month"
        assert any("1000" in a and "12000" in a for a in res.assumptions)
        assert any("conversion check" in c.lower() for c in res.verification_checks)

        # Check executive explanation reflects correct Q* and annual demand
        assert "1,09" in resp.explanation  # matches 1,095 or 1,096
        assert "547.72" in resp.explanation
        assert "12,000" in resp.explanation

    def test_daily_demand_conversion_to_annual(self):
        """Tests that daily demand is annualized using working days (default 250)."""
        agent = InventoryDecisionAgent()
        prompt = "Find EOQ. Demand is 40 units per day, order cost is $50 per order, and holding cost is $2 per unit per year."
        resp = agent.ask(prompt)

        assert resp.success is True
        # 40 units/day * 250 days = 10,000 units/year
        assert resp.extracted_inputs["demand"] == 10000.0
        assert resp.extracted_inputs["stated_demand"] == 40.0
        assert resp.extracted_inputs["demand_period"] == "day"

        res: tools.EOQResult = resp.raw_result
        # Q* = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) approx 707.11
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 707.11
        assert res.recommended_order_quantity in [707, 708]

    def test_weekly_demand_conversion_to_annual(self):
        """Tests that weekly demand is annualized using 52 weeks/year."""
        agent = InventoryDecisionAgent()
        prompt = "Find EOQ. Demand is 200 units per week, order cost is $50 per order, and holding cost is $2 per unit per year."
        resp = agent.ask(prompt)

        assert resp.success is True
        # 200 units/week * 52 weeks = 10,400 units/year
        assert resp.extracted_inputs["demand"] == 10400.0
        assert resp.extracted_inputs["stated_demand"] == 200.0
        assert resp.extracted_inputs["demand_period"] == "week"

        res: tools.EOQResult = resp.raw_result
        # Q* = sqrt(2 * 10400 * 50 / 2) = sqrt(520,000) approx 721.11
        assert pytest.approx(res.optimal_order_quantity, rel=1e-3) == 721.11

    def test_quantity_discount_monthly_demand_conversion(self):
        """Tests that monthly demand is annualized in the quantity discount model."""
        agent = InventoryDecisionAgent()
        prompt = (
            "Demand is 500 units per month, ordering cost is $49, holding cost rate is 20%. "
            "Tiers: 1-999: $5.00, 1000+: $4.80. What is optimal order quantity?"
        )
        resp = agent.ask(prompt)

        assert resp.success is True
        # 500 * 12 = 6,000 units/year
        assert resp.extracted_inputs["demand"] == 6000.0
        assert resp.extracted_inputs["stated_demand"] == 500.0
        assert resp.extracted_inputs["demand_period"] == "month"
        assert resp.raw_result.annual_demand == 6000.0


# ===========================================================================
# 8. Safety Stock & Reorder Point Unit Conversion & BAN 517 Test 7 Regression
# ===========================================================================

class TestSafetyStockROPUnitConversionRegression:
    """
    Automated regression tests verifying that lead time in weeks and months
    is converted to consistent working days with daily demand,
    that decimal lead time standard deviations are extracted accurately,
    and that variable demand and variable lead time (Case 4) are solved
    using sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2).
    """

    def test_professor_report_generator_test7_variable_demand_variable_lead_time(self):
        """
        Exact BAN 517 Project Report Generator Test 7 prompt:
        'A product has average demand of 40 units per day with a standard deviation of 8 units per day.
         Average lead time is 3 weeks with a standard deviation of 0.5 weeks.
         Determine the safety stock and reorder point for a 95% service level.'

        Expected:
            - Daily demand mean d: 40 units/day
            - Daily demand std sigma_d: 8 units/day
            - Stated lead time: 3 weeks -> Converted: 3 * 5 = 15 working days
            - Stated lead time std: 0.5 weeks -> Converted: 0.5 * 5 = 2.5 working days
            - Case: Case 4 (Variable demand, variable lead time)
            - Expected lead time demand mu_L: 40 * 15 = 600 units
            - Demand variance: 15 * 8^2 = 960
            - Lead time variance: 40^2 * 2.5^2 = 1,600 * 6.25 = 10,000
            - Combined std dev sigma_LT: sqrt(960 + 10,000) = sqrt(10,960) approx 104.68997 units
            - Z (95% CSL): 1.64485
            - Safety Stock SS: 1.64485 * 104.68997 approx 172.20 units (rounded up: 173 units)
            - Reorder Point ROP: 600 + 172.20 approx 772.20 units (rounded up: 773 units)
        """
        agent = InventoryDecisionAgent()
        prompt = (
            "A product has average demand of 40 units per day with a standard deviation of 8 units per day. "
            "Average lead time is 3 weeks with a standard deviation of 0.5 weeks. "
            "Determine the safety stock and reorder point for a 95% service level."
        )
        resp = agent.ask(prompt)

        assert resp.success is True, f"Agent failed on Test 7 prompt: {resp.missing_inputs}"
        assert resp.identified_model == "safety_stock_rop"

        # Check extracted inputs
        inputs = resp.extracted_inputs
        assert inputs["demand_mean"] == 40.0
        assert inputs["demand_std"] == 8.0
        assert inputs["demand_unit"] == "day"
        assert inputs["stated_lead_time"] == 3.0
        assert inputs["stated_lead_time_unit"] == "week"
        assert inputs["stated_lead_time_std"] == 0.5
        assert inputs["stated_lead_time_std_unit"] == "week"
        assert inputs["lead_time"] == 15.0
        assert inputs["lead_time_std"] == 2.5
        assert inputs["service_level"] == 0.95

        # Check calculation result
        res: tools.SafetyStockROPResult = resp.raw_result
        assert res.case_description == "Case 4: Variable Demand, Variable Lead Time"
        assert res.has_variable_lead_time is True
        assert res.daily_demand_mean == 40.0
        assert res.daily_demand_std == 8.0
        assert res.lead_time_days == 15.0
        assert res.lead_time_std_days == 2.5
        assert pytest.approx(res.lead_time_demand_mean, rel=1e-3) == 600.0
        assert pytest.approx(res.demand_variance_component, rel=1e-3) == 960.0
        assert pytest.approx(res.lead_time_variance_component, rel=1e-3) == 10000.0
        assert pytest.approx(res.lead_time_demand_std, rel=1e-3) == 104.690

        # Check Z and Service Level
        assert pytest.approx(res.service_level, rel=1e-3) == 0.95
        assert pytest.approx(res.z_score, rel=1e-3) == 1.6449

        # Check Safety Stock and ROP (both exact float and integer ceiling)
        assert pytest.approx(res.safety_stock, rel=1e-3) == 172.20
        assert res.recommended_safety_stock == 173
        assert pytest.approx(res.reorder_point, rel=1e-3) == 772.20
        assert res.recommended_reorder_point == 773
        assert res.final_answer == 773

        # Check formula and substitutions
        assert "sqrt(15 * (8)^2 + (40)^2 * (2.5)^2)" in res.formula_substituted
        assert "104.6900" in res.formula_substituted
        assert "172.20" in res.formula_substituted
        assert "772.20" in res.formula_substituted

        # Check verifiable step components include stated values and conversion note
        assert res.intermediate_calculations.get("stated_lead_time") == 3.0
        assert res.intermediate_calculations.get("stated_lead_time_unit") == "week"
        assert res.intermediate_calculations.get("effective_lead_time_days") == 15.0
        assert res.intermediate_calculations.get("stated_lead_time_std") == 0.5
        assert res.intermediate_calculations.get("effective_lead_time_std_days") == 2.5
        assert any("3 weeks" in a and "15" in a for a in res.assumptions)
        assert any("conversion verified" in c.lower() for c in res.verification_checks)

        # Check executive explanation reflects correct SS and ROP
        assert "173" in resp.explanation
        assert "773" in resp.explanation
        assert "172.20" in resp.explanation
        assert "772.20" in resp.explanation

    def test_constant_lead_time_in_weeks_conversion(self):
        """Tests that constant lead time in weeks is converted to days without variability."""
        agent = InventoryDecisionAgent()
        prompt = (
            "Average daily demand is 50 units with standard deviation of 10 units. "
            "Lead time is 2 weeks (constant). 95% service level."
        )
        resp = agent.ask(prompt)

        assert resp.success is True
        # 2 weeks * 5 days/week = 10 days
        assert resp.extracted_inputs["lead_time"] == 10.0
        assert resp.extracted_inputs["lead_time_std"] == 0.0
        assert resp.raw_result.case_description == "Case 2: Variable Demand, Constant Lead Time"
        # mu_L = 50 * 10 = 500, sigma_LT = 10 * sqrt(10) = 31.62277
        assert pytest.approx(resp.raw_result.lead_time_demand_mean, rel=1e-3) == 500.0
        assert pytest.approx(resp.raw_result.lead_time_demand_std, rel=1e-3) == 31.623
        # SS = 1.6449 * 31.623 = 52.01 -> 53 units
        assert pytest.approx(resp.raw_result.safety_stock, rel=1e-2) == 52.01
        assert resp.raw_result.recommended_safety_stock in [52, 53]

    def test_lead_time_in_months_conversion(self):
        """Tests that lead time specified in months is converted to working days."""
        agent = InventoryDecisionAgent()
        prompt = (
            "Daily demand is 20 units with standard deviation of 4 units. "
            "Supplier lead time is 1 month with standard deviation of 0.2 months. 95% service level."
        )
        resp = agent.ask(prompt)

        assert resp.success is True
        # 250 / 12 = 20.8333 working days/month
        expected_days = 250.0 / 12.0
        assert pytest.approx(resp.extracted_inputs["lead_time"], rel=1e-3) == expected_days
        assert pytest.approx(resp.extracted_inputs["lead_time_std"], rel=1e-3) == 0.2 * expected_days
        assert resp.raw_result.case_description == "Case 4: Variable Demand, Variable Lead Time"



