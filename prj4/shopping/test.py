import unittest
from shopping import load_data

class TestShopping(unittest.TestCase):
    def test_load_data(self):
        evidence, labels = load_data("shopping.csv")
        for ev, lab in zip(evidence, labels):
            types = [type(data) for data in ev]
            self.assertEqual(types, [int, float, int, float, int, float,
                                    float, float, float, float, int, int,
                                    int, int, int, int, int])
            self.assertEqual(type(lab), int)
        
if __name__ == "__main__":
    unittest.main()