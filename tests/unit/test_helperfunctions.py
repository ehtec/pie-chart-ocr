import unittest
from piechartocr import helperfunctions


class TestBaseFunctions(unittest.TestCase):

    def test_remove_sc_prefix(self):

        word = helperfunctions.remove_sc_prefix('@;.T3st1n')
        word1 = 'T3st1n'
        self.assertEqual(word, word1)
