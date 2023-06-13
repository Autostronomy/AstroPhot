<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/AP_logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset="media/AP_logo.png">
  <img alt="AutoPhot logo" src="media/AP_logo.png" width="70%">
</picture>


[![unittests](https://github.com/Autostronomy/AutoPhot/actions/workflows/testing.yaml/badge.svg?branch=main)](https://github.com/Autostronomy/AutoPhot/actions/workflows/testing.yaml)
[![docs](https://github.com/Autostronomy/AutoPhot/actions/workflows/documentation.yaml/badge.svg?branch=main)](https://autostronomy.github.io/AutoPhot/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi](https://img.shields.io/pypi/v/autophot.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/autophot/)
[![downloads](https://img.shields.io/pypi/dm/autophot?label=PyPI%20Downloads)](https://libraries.io/pypi/autophot)
[![codecov](https://img.shields.io/codecov/c/github/Autostronomy/AutoPhot?logo=codecov)](https://app.codecov.io/gh/Autostronomy/AutoPhot?search=&displayType=list)

AutoPhot is a fast, flexible, and automated astronomical image modelling tool for precise parallel multi-wavelength photometry. It is a python based package that uses PyTorch to quickly and efficiently perform analysis tasks. Written by [Connor Stone](https://connorjstone.com/) for tasks such as LSB imaging, handling crowded fields, multi-band photometry, and analyzing massive data from future telescopes. AutoPhot is flexible and fast for any astronomical image modelling task. While it uses PyTorch (originally developed for Machine Learning) it is NOT a machine learning based tool.

## Installation

AutoPhot can be installed with pip:

```
pip install autophot
```

If PyTorch gives you any trouble on your system, just follow the instructions on the [pytorch website](https://pytorch.org/) to install a version for your system.

Also note that AutoPhot is only available for python3.

See [the documentation](https://autostronomy.github.io/AutoPhot/) for more details.

## Documentation

You can find the documentation at the [GitHub Pages site connected with the AutoPhot project](https://autostronomy.github.io/AutoPhot/) which covers many of the main use cases for AutoPhot. It is still in development, but lots of useful information is there. Feel free to contact the author, [Connor Stone](https://connorjstone.com/), for any questions not answered by the documentation or tutorials.

## Credit / Citation

If you use AutoPhot in your research, please follow the [citation instructions here](https://autostronomy.github.io/AutoPhot/citation.html). A new paper for the updated AutoPhot code is in the works.
