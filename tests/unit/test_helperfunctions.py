import unittest
import unittest.mock
import os
import cv2
import random
import shutil
from piechartocr import helperfunctions
import sys
import logging
import io


# delta for color tests
COLOR_DELTA = 2.0

logger = logging.getLogger()
logger.level = logging.DEBUG


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

    def test_group_pairs_to_nested_list(self):
        # random test
        edges = [random.sample(range(0, 100), 2) for _ in range(1, 70)]
        print(edges)
        components = helperfunctions.group_pairs_to_nested_list(edges)
        print(components)
        for edge in edges:
            occ_first_vert = 0
            occ_second_vert = 0
            for comp in components:
                f = comp.count(edge[0])
                s = comp.count(edge[1])
                self.assertTrue((s > 0 and f > 0) or (s == 0 and f == 0))
                occ_first_vert += f
                occ_second_vert += s
            self.assertEqual(occ_first_vert, 1)
            self.assertEqual(occ_second_vert, 1)

    def test_pre_rectangle_center(self):
        p1 = [0.0, 5, 2.2, 6]
        x, y = helperfunctions.pre_rectangle_center(p1)
        self.assertAlmostEqual(x, 1.1)
        self.assertAlmostEqual(y, 5.5)

        p1 = [-3.5, 3.28, 8, 2390.6]
        x, y = helperfunctions.pre_rectangle_center(p1)
        self.assertAlmostEqual(x, 2.25)
        self.assertAlmostEqual(y, 1196.94)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, path, expected_output, mock_stdout):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        try:
            helperfunctions.clean_folder_contents(path)

        finally:
            logger.removeHandler(stream_handler)
        self.assertTrue(expected_output in mock_stdout.getvalue())

    def test_clean_folder_contents(self):

        # create garbage
        path = os.path.join(helperfunctions.get_root_path(), 'data', "test_folder")  # ! also using get_root_path
        if not os.path.isdir(path):
            os.mkdir(path)
        folder1 = os.path.join(path, "1")
        if not os.path.isdir(folder1):
            os.mkdir(folder1)
        os.chmod(folder1, 0o777)
        folder2 = os.path.join(path, "2")
        if not os.path.isdir(folder2):
            os.mkdir(folder2)
        folder1_1 = os.path.join(folder1, "1_1")
        if not os.path.isdir(folder1_1):
            os.mkdir(folder1_1)

        textfile1 = os.path.join(path, "trash1.txt")
        trash1 = open(textfile1, "w+")
        trash1.write("this is basic garbage")
        trash1.close()

        textfile2 = os.path.join(folder2, "trash2.txt")
        trash2 = open(textfile2, "w+")
        trash2.write("this is level 2 garbage")
        trash2.close()

        textfile3 = os.path.join(folder1_1, "trash3.txt")
        trash3 = open(textfile3, "w+")
        trash3.write("this is level 3 garbage")
        trash3.close()

        # removing permissions
        os.chmod(folder1, 0o000)

        # attempt clean without permissions
        self.assert_stdout(path, "PermissionError")

        # giving permissions back
        os.chmod(folder1, 0o777)

        # clean
        helperfunctions.clean_folder_contents(path)

        # inspect results
        self.assertTrue(not os.path.isdir(folder1) and not os.path.isdir(folder2) and not os.path.isdir(textfile1))

        # remove test folder
        shutil.rmtree(path)
