[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)   
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=logo=scikit-learn&logoColor=white)

# OVERVIEW

This folder is going to contain any scikit-learn tutorials or mini-projects that
I play around with.

Other README.md files have been created within sub-directories to keep some basic notes
on what has been done.

## Code Formatting
---

Basic code formatting is provided using the pyproject.toml and .flake8 configuration files
and can be run as follows:

```
black .
isort .
flake8 .
```

## Setup
---

Within the parent directory, a requirements file has been created to install the
necessary packages.   A new virtual environment should be created and activated
using this requirements file.   eg:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
