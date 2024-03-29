from setuptools import setup, Extension, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# version parameters
MAJOR_VERSION = "0"
MINOR_VERSION = "6"
SUB_MINOR_VERSION = "6"


# version of the package
# VERSION = "0.5.7"
VERSION = f"{MAJOR_VERSION}.{MINOR_VERSION}.{SUB_MINOR_VERSION}"


colorprocesser_module = Extension(
    'libcolorprocesser',
    sources=['src/colorprocesser.cpp'],
    define_macros=[('MAJOR_VERSION', MAJOR_VERSION), ('MINOR_VERSION', MINOR_VERSION)],
    # version=VERSION,
    include_dirs=['color/src'],
    libraries=[],
    library_dirs=['/usr/local/lib']
)

polygoncalc_module = Extension(
    'libpolygoncalc',
    sources=['src/polygoncalc.cpp'],
    define_macros=[('MAJOR_VERSION', MAJOR_VERSION), ('MINOR_VERSION', MINOR_VERSION)],
    # version=VERSION,
    include_dirs=[],
    libraries=['boost_system'],
    library_dirs=['/usr/local/lib']
)

setup(name="piechartocr",
      packages=find_packages(),  # ['piechartocr'],
      version=VERSION,
      license="MIT",
      description="Pie Chart Optical Character Recognition",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author="Elias Hohl",
      author_email="elias.hohl@ehtec.co",
      url="https://git.ehtec.co/research/pie-chart-ocr",
      download_url=f"https://git.ehtec.co/research/pie-chart-ocr/-/archive/v{VERSION}-beta/pie-chart-ocr-v{VERSION}-beta.zip",
      keywords="pie chart parsing ocr",
      install_requires=[
          "Cython",
          "cvxpy",
          "cvxopt",
          "scipy",
          "numpy",
          "pytesseract",
          "pillow",
          "opencv-python",
          "opencv-contrib-python",
          "scikit-image",
          "ellipsefitting",
          "matplotlib",
          "shapely",
          "tqdm",
          "colormath",
          "colorthief",
          "scikit-learn",
          "nose2",
          "coverage",
          "nltk"
      ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      ext_modules=[colorprocesser_module, polygoncalc_module])
