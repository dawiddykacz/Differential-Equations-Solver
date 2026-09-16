import unittest

from HeatmapScript import parse_name


class EquationsTest(unittest.TestCase):
    def test_1(self):
        expected = {
            "name": "1 example problem loss",
            "alpha": 0.9,
            "betta": None,
            "weight": 1,
            "is_wang_dynamic_weight": True,
        }
        text = "1 example problem loss (wang and weight_conditions 1, weight_data 1) with noise = False alpha = 0.9"

        self.assertEqual(parse_name(text), expected)

    def test_2(self):
        expected = {
            "name": "1 example problem loss",
            "alpha": None,
            "betta": None,
            "weight": 1,
            "is_wang_dynamic_weight": False,
        }
        text = "1 example problem loss with noise False weight_conditions 1, weight_data 1"

        self.assertEqual(parse_name(text), expected)
