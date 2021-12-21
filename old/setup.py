from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("pie_chart_ocr.pyx", language_level=3)
)
