# pie-chart-ocr
A tool to extract tabular data from pie charts, developed as a component of the Finomena toolkit.

Get more information here:

https://finminity.com

https://analytics.finminity.finance

https://t.me/FinminityOfficial

Note: The original repository was moved to https://git.ehtec.co/research/pie-chart-ocr.
https://github.com/ehtec/pie-chart-ocr is a mirror.

# Installation

Install Boost and Tesseract:

`sudo apt install libboost-all-dev tesseract-ocr build-essential`

Clone this repository including submodules:

```commandline
git clone --recursive https://github.com/ehtec/pie-chart-ocr.git
cd pie-chart-ocr
```

Install Python requirements:

```commandline
python3 -m pip install -r requirements.txt
```

Compile libraries:

```commandline
python3 setup.py build_ext
mkdir lib
sudo python3 setup.py install --install-lib lib/
```

Eventually change the `MAX_WORKERS` number in `pie_chart_ocr.py`.

Create temporary directories:
```commandline
mkdir temp
mkdir temp1
mkdir temp2
```

Unpack test charts:

```commandline
cd data
unzip charts_steph.zip
cd ..
```

# Usage

Run tests on an image for the first time (also uses upsampling):

```commandline
python3 test_superreshelper.py
```

The script asks for an image ID. Use `4`, for example.

Run tests for images that were already upsampled:

```commandline
python3 test_pie_chart_ocr.py
```
