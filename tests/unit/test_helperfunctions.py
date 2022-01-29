import unittest
import os
import cv2
from piechartocr import helperfunctions


# delta for color tests
COLOR_DELTA = 2.0


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

    def test_get_cv2_dominant_color(self):

        image_path = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_1.png')
        image_path1 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_2.jpg')
        image_path2 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'image-019_1.png')
        image_path3 = os.path.join(helperfunctions.get_root_path(), 'test_data', '262626.png')
        dominant_color = helperfunctions.get_cv2_dominant_color(cv2.imread(image_path), 5)
        dominant_color1 = helperfunctions.get_cv2_dominant_color(cv2.imread(image_path1), 5)
        dominant_color2 = helperfunctions.get_cv2_dominant_color(cv2.imread(image_path2), 5)
        dominant_color3 = helperfunctions.get_cv2_dominant_color(cv2.imread(image_path3), 5)
        for rgb1, rgb2 in zip(dominant_color, (255, 255, 255)):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color1, (255, 255, 255)):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color2, (255, 255, 255)):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color3, (51, 51, 51)):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)

    def test_get_cv2_dominant_color_3(self):

        image_path = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_1.png')
        image_path1 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_2.jpg')
        image_path2 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'image-019_1.png')
        image_path3 = os.path.join(helperfunctions.get_root_path(), 'test_data', '262626.png')
        dominant_color = helperfunctions.get_cv2_dominant_color_3(cv2.imread(image_path), 5)
        dominant_color1 = helperfunctions.get_cv2_dominant_color_3(cv2.imread(image_path1), 5)
        dominant_color2 = helperfunctions.get_cv2_dominant_color_3(cv2.imread(image_path2), 5)
        dominant_color3 = helperfunctions.get_cv2_dominant_color_3(cv2.imread(image_path3), 5)
        for rgb1, rgb2 in zip(dominant_color, [254, 254, 254]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color1, [253, 254, 254]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color2, [254, 254, 254]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color3, [38, 38, 38]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)

    def test_get_cv2_dominant_color_5(self):

        image_path = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_1.png')
        image_path1 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'test_2.jpg')
        image_path2 = os.path.join(helperfunctions.get_root_path(), 'test_data', 'image-019_1.png')
        image_path3 = os.path.join(helperfunctions.get_root_path(), 'test_data', '262626.png')
        dominant_color = helperfunctions.get_cv2_dominant_color_5(cv2.imread(image_path), 5)
        dominant_color1 = helperfunctions.get_cv2_dominant_color_5(cv2.imread(image_path1), 5)
        dominant_color2 = helperfunctions.get_cv2_dominant_color_5(cv2.imread(image_path2), 5)
        dominant_color3 = helperfunctions.get_cv2_dominant_color_5(cv2.imread(image_path3), 5)
        for rgb1, rgb2 in zip(dominant_color, [196, 218, 215]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color1, [232, 235, 230]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color2, [231, 249, 219]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
        for rgb1, rgb2 in zip(dominant_color3, [38, 38, 38]):
            self.assertAlmostEqual(rgb1, rgb2, delta=COLOR_DELTA)
