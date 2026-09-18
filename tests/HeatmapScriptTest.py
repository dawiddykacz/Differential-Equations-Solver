import unittest

from HeatmapScript import parse_name


class EquationsTest(unittest.TestCase):
    def test_1(self):
        expected = {
            "name": "1 example problem loss",
            "alpha": 0.9,
            'is_wang_dynamic_weight': True,
            "betta": None,
            'weight_conditions': 10000.0,
            'weight_data': 10000.0,
            'weight_pde': 1.0,
            'weights_combo': 'pde: 1\ncond: 10000\ndata: 10000'
        }
        text = ("1 example problem loss (wang and weight_pde 1 weight_conditions 10000, weight_data 10000) with "
                "noise = False alpha = 0.9")

        self.assertEqual(parse_name(text), expected)

    def test_2(self):
        expected = {
            'is_wang_dynamic_weight': False,
            'name': '1 example simple problem loss',
            "alpha": None,
            "betta": None,
            'weight_conditions': None,
            'weight_data': 1.0,
            'weight_pde': 0.0001,
            'weights_combo': 'pde: 0.0001\ndata: 1'
        }
        text = "1 example simple problem loss with noise False, weight_pde 0.0001 weight_data 1"

        self.assertEqual(parse_name(text), expected)
