import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from T0_01_data_loader import main as build_t0_config
from T1_01_population_forecast import forecast_population_5years, load_t0_config
from T1_02_theoretical_demand import (
    compute_theoretical_demand,
    compute_theoretical_demand_by_elderly_type,
)
from T1_03_constrained_demand import (
    apply_consumption_constraint,
    compute_reduction_coefficient,
    compute_spending_limit_per_person,
    compute_theoretical_expense_per_person,
    convert_monthly_to_daily,
)


class T1Q1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):
            build_t0_config()
        cls.config = load_t0_config()
        cls.history, cls.population_year5 = forecast_population_5years(cls.config)

    def test_population_forecast_keeps_float_values(self):
        self.assertEqual(self.population_year5.dtype.kind, 'f')
        self.assertFalse(np.allclose(self.population_year5, np.rint(self.population_year5)))

    def test_population_forecast_shapes(self):
        self.assertEqual(len(self.history), 5)
        self.assertEqual(self.population_year5.shape, (10, 3))
        self.assertTrue(np.all(self.population_year5 >= 0))

    def test_theoretical_demand_shapes(self):
        demand = compute_theoretical_demand(
            self.population_year5,
            self.config['demand_rate_matrix']
        )
        demand_by_type = compute_theoretical_demand_by_elderly_type(
            self.population_year5,
            self.config['demand_rate_matrix']
        )
        self.assertEqual(demand.shape, (10, 6))
        self.assertTrue(np.allclose(demand, demand_by_type['total']))

    def test_reduction_coefficient_range(self):
        spending_limits = compute_spending_limit_per_person(
            self.config['monthly_income_per_capita'],
            self.config['spending_cap_ratios']
        )
        expense = compute_theoretical_expense_per_person(
            self.config['demand_rate_matrix'],
            self.config['service_prices']
        )
        coefficient = compute_reduction_coefficient(spending_limits, expense)
        self.assertEqual(coefficient.shape, (10, 3))
        self.assertTrue(np.all((coefficient >= 0) & (coefficient <= 1)))

    def test_constrained_demand_not_exceed_theoretical(self):
        theoretical = compute_theoretical_demand(
            self.population_year5,
            self.config['demand_rate_matrix']
        )
        spending_limits = compute_spending_limit_per_person(
            self.config['monthly_income_per_capita'],
            self.config['spending_cap_ratios']
        )
        expense = compute_theoretical_expense_per_person(
            self.config['demand_rate_matrix'],
            self.config['service_prices']
        )
        coefficient = compute_reduction_coefficient(spending_limits, expense)
        constrained = apply_consumption_constraint(
            self.population_year5,
            self.config['demand_rate_matrix'],
            coefficient
        )['total']
        self.assertTrue(np.all(constrained <= theoretical + 1e-8))

    def test_daily_demand_conversion(self):
        monthly = np.array([[30.0, 90.0], [15.0, 45.0]])
        daily = convert_monthly_to_daily(monthly, 30)
        self.assertTrue(np.allclose(daily, [[1.0, 3.0], [0.5, 1.5]]))


if __name__ == '__main__':
    unittest.main()
