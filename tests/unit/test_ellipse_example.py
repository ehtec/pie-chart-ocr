import unittest
import numpy as np
from piechartocr import ellipse_example


class TestEllipseExample(unittest.TestCase):

    def test_make_test_ellipse(self):

        x = ellipse_example.make_test_ellipse([1, 2, 3])

        self.assertEqual(len(x), 2)
        self.assertIsInstance(x, list)
        for i in x:
            self.assertIsInstance(i, np.ndarray)
            self.assertEqual(i.shape, (1000, ))
