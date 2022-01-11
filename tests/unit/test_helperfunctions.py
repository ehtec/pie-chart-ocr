import unittest
from piechartocr import helperfunctions


class TestBaseFunctions(unittest.TestCase):

    def test_remove_sc_prefix(self):

        word = helperfunctions.remove_sc_prefix('@;.T3st1n')
        special_characters = r'[A-Za-z0-9].*'
        self.assertEqual(word, special_characters)
