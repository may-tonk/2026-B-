import pickle
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
from T0_validation_utils import validate_config


class T0ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):
            build_t0_config()

        config_path = PROJECT_ROOT / 'model' / 'T0_01_config.pkl'
        with config_path.open('rb') as config_file:
            cls.config = pickle.load(config_file)

    def test_standard_config_passes_validation(self):
        validate_config(self.config)

    def test_coverage_mask_matches_distance_threshold(self):
        expected_mask = (self.config['distance_matrix'] <= 1000).astype(int)
        self.assertTrue(np.array_equal(self.config['coverage_mask'], expected_mask))

    def test_spending_caps_follow_attachment_rows(self):
        self.assertTrue(np.allclose(self.config['spending_cap_ratios'], [0.20, 0.25, 0.30]))

    def test_station_specs_include_attachment_scales(self):
        self.assertEqual(set(self.config['station_specs']), {'small', 'medium', 'large'})

    def test_satisfaction_rules_keep_attachment_segments(self):
        rules = self.config['satisfaction_rules']
        self.assertEqual(len(rules['distance']['segments']), 4)
        self.assertEqual(len(rules['response']['segments']), 5)
        self.assertEqual(len(rules['price']['segments']), 4)


if __name__ == '__main__':
    unittest.main()
