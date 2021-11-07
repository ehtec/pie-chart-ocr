# pie-chart-ocr
A tool to extract tabular data from pie charts, developed as a component of the Finomena toolkit.

Get more information here:

https://finminity.com

https://analytics.finminity.finance

https://t.me/FinminityOfficial

# Installation

Install Boost and Tesseract:

`sudo apt install libboost-all-dev tesseract-ocr`

Install Python requirements:

`python3 -m pip install -r requirements.txt`

Compile libraries:

`bash compile_polygoncalc.sh`

`bash compile_colorprocesser.sh`

Eventually change the `MAX_WORKERS` number in `pie_chart_ocr.pyx`

Build Cython modules:

`python3 setup.py build_ext --inplace`

# Usage

Run tests:

`python3 test_pie_chart_ocr.py`
