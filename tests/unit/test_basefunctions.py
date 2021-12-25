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
