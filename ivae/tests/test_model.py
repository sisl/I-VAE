import unittest
from ivae.model import IVAE

class TestModel(unittest.TestCase):
    def test_model(self):
        model = IVAE(1, 1, 32, 0.01)
        return True

if __name__ == "__main__":
    unittest.main()
