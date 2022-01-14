import unittest
from piechartocr import helperfunctions


class TestHelperFunctions(unittest.TestCase):

    def test_remove_sc_prefix(self):

        word = helperfunctions.remove_sc_prefix('@Testing')
        word1 = 'Testing'
        self.assertEqual(word, word1)

    def test_remove_sc_suffix(self):

        word = helperfunctions.remove_sc_suffix('Testing@')
        word1 = 'Testing'
        self.assertEqual(word, word1)
