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

```commandline
sudo apt install libboost-system-dev tesseract-ocr build-essential git
```

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
unzip data/charts_steph.zip -d data
```

# Usage

Run unit tests:

```commandline
python3 -m nose2 --start-dir tests/ --with-coverage
```

Run legacy tests / examples:

```commandline
python3 run_examples.py
```

The script asks for an image ID. Use `4`, for example.

You need to close all `matplotlib` figures by pressing the "x", and all `opencv` images
labeled `img` or `vis` by pressing an arbitrary key.
