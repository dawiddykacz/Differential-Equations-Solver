import unittest

from bin.Compile_plots import get_name_mapped


class EquationsTest(unittest.TestCase):
    def test_1(self):
        self.assertEqual(get_name_mapped("2 example simple problem loss  with noise False, weight_pde 1 weight_data 1"),
                         "2.1.a")
        self.assertEqual(get_name_mapped("1 example simple problem loss  with noise False, weight_pde 1 weight_data 1"),
                         "1.1.a")
        self.assertEqual(get_name_mapped("1 example problem loss  with noise False, weight_pde 1 weight_data 1"),
                         "1.2.a")
        self.assertEqual(get_name_mapped("2 example problem loss  with noise False, weight_pde 1 weight_data 1"),
                         "2.2.a")

        self.assertEqual(get_name_mapped("1 example simple problem loss (wang and weight_data 1 weight_pde 1) "
                                         "with noise = False alpha = 0.9"), "1.1.c")
        self.assertEqual(get_name_mapped("1 example simple problem loss (wang and weight_data 10000 weight_pde 1) "
                                         "with noise = False alpha = 0.9"), "1.1.d2")
        self.assertEqual(get_name_mapped("1 example simple problem loss (wang and weight_data 1 weight_pde 0.0001) "
                                         "with noise = False alpha = 0.9"), "1.1.d1")
        self.assertEqual(get_name_mapped("1 example simple problem loss with noise False, weight_pde 0.0001 "
                                         "weight_data 1"), "1.1.b1")
        self.assertEqual(
            get_name_mapped("1 example simple problem loss with noise False, weight_pde 1 weight_data 10000"),
            "1.1.b2")
