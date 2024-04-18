<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Autostronomy/AstroPhot/blob/main/media/AP_logo_white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Autostronomy/AstroPhot/blob/main/media/AP_logo.png?raw=true">
  <img alt="AstroPhot logo" src="media/AP_logo.png" width="70%">
</picture>

[![unittests](https://github.com/Autostronomy/AstroPhot/actions/workflows/testing.yaml/badge.svg?branch=main)](https://github.com/Autostronomy/AstroPhot/actions/workflows/testing.yaml)
[![Documentation Status](https://readthedocs.org/projects/astrophot/badge/?version=latest)](https://astrophot.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Autostronomy/AstroPhot/main.svg)](https://results.pre-commit.ci/latest/github/Autostronomy/AstroPhot/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi](https://img.shields.io/pypi/v/astrophot.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/astrophot/)
[![downloads](https://img.shields.io/pypi/dm/astrophot?label=PyPI%20Downloads)](https://libraries.io/pypi/astrophot)
[![codecov](https://img.shields.io/codecov/c/github/Autostronomy/AstroPhot?logo=codecov)](https://app.codecov.io/gh/Autostronomy/AstroPhot?search=&displayType=list)
[![Static Badge](https://img.shields.io/badge/ADS-record-2A79E4)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.6377S/abstract)
[![DOI](https://zenodo.org/badge/473209170.svg)](https://zenodo.org/doi/10.5281/zenodo.10798979)

AstroPhot is a fast, flexible, and automated astronomical image modelling tool
for precise parallel multi-wavelength photometry. It is a python based package
that uses PyTorch to quickly and efficiently perform analysis tasks. Written by
[Connor Stone](https://connorjstone.com/) for tasks such as LSB imaging,
handling crowded fields, multi-band photometry, and analyzing massive data from
future telescopes. AstroPhot is flexible and fast for any astronomical image
modelling task. While it uses PyTorch (originally developed for Machine
Learning) it is NOT a machine learning based tool.

## Installation

AstroPhot can be installed with pip:

```
pip install astrophot
```

If PyTorch gives you any trouble on your system, just follow the instructions on
the [pytorch website](https://pytorch.org/) to install a version for your
system.

Also note that AstroPhot is only available for python3.

See [the documentation](https://astrophot.readthedocs.io) for more details.

## Documentation

You can find the documentation at the
[ReadTheDocs site connected with the AstroPhot project](https://astrophot.readthedocs.io)
which covers many of the main use cases for AstroPhot. It is still in
development, but lots of useful information is there. Feel free to contact the
author, [Connor Stone](https://connorjstone.com/), for any questions not
answered by the documentation or tutorials.

## Credit / Citation

If you use AstroPhot in your research, please follow the
[citation instructions here](https://autostronomy.github.io/AstroPhot/citation.html).

## Thanks to our contributors!

[![Contributors](https://contrib.rocks/image?repo=Autostronomy/AstroPhot)](https://github.com/Autostronomy/AstroPhot/graphs/contributors)
