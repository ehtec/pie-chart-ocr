import unittest
from piechartocr import basefunctions
import os


class TestBaseFunctions(unittest.TestCase):

    def test_get_root_path(self):

        path = basefunctions.get_root_path()

        # check for the tox.ini file to see if we indeed are in the root directory
        file_path = os.path.join(path, 'tox.ini')
        self.assertTrue(os.path.isfile(file_path))

        # check for the LICENSE.txt file to see if we indeed are in the root directory
        file_path = os.path.join(path, 'LICENSE.txt')
        self.assertTrue(os.path.isfile(file_path))

        # check for the piechartocr folder to see if we indeed are in the root directory
        folder_path = os.path.join(path, 'piechartocr')
        self.assertTrue(os.path.isdir(folder_path))

    def test_complex_to_real(self):

        first = basefunctions.complex_to_real(complex(3, 0))
        second = 3.0
        self.assertAlmostEqual(first, second)

        first_num = basefunctions.complex_to_real(2.0)
        second_num = 2.0
        self.assertAlmostEqual(first_num, second_num)

        with self.assertRaisesRegex(ValueError, 'Imaginary part not zero'):

            basefunctions.complex_to_real(complex(3, 3))

    def test_find_lib(self):

        lib_path = basefunctions.find_lib(os.path.join(basefunctions.get_root_path()), 'libcolorprocesser')
        self.assertIn('libcolorprocesser', lib_path)

        lib_path = basefunctions.find_lib(os.path.join('searchpath'), 'lib')
        self.assertIsNone(lib_path)

        file_path = basefunctions.find_lib(os.path.join(basefunctions.get_root_path()), ('piechartocr'))
        self.assertIsNone(file_path)
