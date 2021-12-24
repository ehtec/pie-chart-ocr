from distutils.core import setup, Extension


colorprocesser_module = Extension(
    'libcolorprocesser',
    sources=['src/colorprocesser.cpp'],
    define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', 3)],
    include_dirs=['color/src'],
    libraries=[],
    library_dirs=['/usr/local/lib']
)

setup(name="PieChartOCR",
      version="0.3",
      description="Pie Chart Optical Character Recognition",
      ext_modules=[colorprocesser_module])
