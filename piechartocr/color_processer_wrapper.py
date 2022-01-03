from ctypes import cdll, c_double
# from ctypes.util import find_library
import ctypes
# from numpy.ctypeslib import as_ctypes, as_array, as_ctypes_type
import numpy as np
from numpy.ctypeslib import ndpointer
import copy
import logging
from .basefunctions import find_lib, get_root_path
import os


# relative path to colorprocesser library
RELATIVE_LIBRARY_PATH = ""
# RELATIVE_LIBRARY_PATH = "build/"


# lib_path = find_library('colorprocesser')
lib_path = find_lib(os.path.join(get_root_path(), RELATIVE_LIBRARY_PATH), 'libcolorprocesser')
if not bool(lib_path):
    raise FileNotFoundError("colorprocesser library not found!")
# lib_full_path = os.path.join(get_root_path(), RELATIVE_LIBRARY_PATH, lib_path)
lib_full_path = lib_path
logging.info("colorprocesser library path: {0}".format(lib_full_path))
lib = cdll.LoadLibrary(lib_full_path)
# set output types for ColorProcesser methods
lib.ColorProcesser_helloworld.restype = c_double
lib.ColorProcesser_test_calc.restype = c_double

lib.ColorProcesser_color_distance.restype = c_double


class ColorProcesser(object):
    def __init__(self):
        self.obj = lib.ColorProcesser_new()
    #     fun = lib.ColorProcesser_new
    #     fun.argtypes = []
    #     fun.restype = ctypes.c_void_p
    #     self.obj = fun()
    #
    # def __del__(self):
    #     fun = lib.ColorProcesser_delete
    #     fun.argtypes = [ctypes.c_void_p]
    #     fun.restype = None
    #     fun(self.obj)

    def helloworld(self):
        return lib.ColorProcesser_helloworld(self.obj)

    def test_calc(self):
        return lib.ColorProcesser_test_calc(self.obj)

    def color_distance(self, c1, c2):

        return lib.ColorProcesser_color_distance(
            self.obj,
            ctypes.c_double(c1[0]),
            ctypes.c_double(c1[1]),
            ctypes.c_double(c1[2]),
            ctypes.c_double(c2[0]),
            ctypes.c_double(c2[1]),
            ctypes.c_double(c2[2])
        )

    def array_color_distance(self, the_color, color_array):

        input_shape = color_array.shape

        # print(input_shape)

        m = np.prod(input_shape[:-1])

        # print(m)

        new_color_array = color_array.reshape(m, 3)

        # print(new_color_array)

        r1 = the_color[0]
        g1 = the_color[1]
        b1 = the_color[2]

        r2 = np.ascontiguousarray(new_color_array[:, 0], dtype=np.double)
        g2 = np.ascontiguousarray(new_color_array[:, 1], dtype=np.double)
        b2 = np.ascontiguousarray(new_color_array[:, 2], dtype=np.double)

        # the_color_distances = np.ascontiguousarray(np.zeros((m,)), dtype=np.double)

        lib.ColorProcesser_array_color_distance.restype = ndpointer(dtype=c_double, shape=(m, ))

        res_array = lib.ColorProcesser_array_color_distance(
            self.obj,
            c_double(r1),
            c_double(g1),
            c_double(b1),
            ctypes.c_void_p(r2.ctypes.data),
            ctypes.c_void_p(g2.ctypes.data),
            ctypes.c_void_p(b2.ctypes.data),
            # ctypes.c_void_p(the_color_distances.ctypes.data),
            ctypes.c_int(m)
        )

        # res_array_content = ctypes.cast(res_array, ndpointer(dtype=c_double, shape=(m, )))

        res_array_copy = copy.deepcopy(res_array)

        # res_array_p = ctypes.cast(res_array, ctypes.POINTER(c_double))

        res_array_p = res_array.ctypes.data_as(ctypes.POINTER(c_double))

        lib.free_double_array(res_array_p)

        return res_array_copy.reshape(input_shape[:-1])
