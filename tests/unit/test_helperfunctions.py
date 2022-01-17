import unittest
import os
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

    def test_rect_from_pre(self):

        p1 = helperfunctions.rect_from_pre((0, 2, 1, 0))
        p2 = ((0, 2), (0, 0), (1, 0), (1, 2))
        self.assertEqual(p1, p2)

    def test_hash_file(self):

        path = helperfunctions.get_root_path()
        file_hash = helperfunctions.hash_file(os.path.join(path, 'test_data', 'tox.ini'))
        self.assertEqual(file_hash, '918f4920ab0b03269643a62cd017e35b0dc4ac5fdc3b9486a0f8c2b7b71eb7d2')

    def test_integerize(self):

        list1 = helperfunctions.integerize([2.0, 3.0])
        list2 = ([2, 3])
        self.assertEqual(list1, list2)
        for i in list1:
            self.assertIsInstance(i, int)
