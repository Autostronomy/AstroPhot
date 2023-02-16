<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/AP_logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset="media/AP_logo.png">
  <img alt="AutoProf logo" src="media/AP_logo.png" width="70%">
</picture>


![unittests](https://github.com/ConnorStoneAstro/AutoProf/actions/workflows/testing.yaml/badge.svg?branch=main)
![docs](https://github.com/ConnorStoneAstro/AutoProf/actions/workflows/documentation.yaml/badge.svg?branch=main)
![pypi](https://img.shields.io/pypi/v/autoprof.svg?logo=pypi&logoColor=white&label=PyPI)
![downloads](https://img.shields.io/pypi/dm/autoprof?label=PyPI%20Downloads)


AutoProf is a fast, flexible, and automated astronomical image modelling tool for precise parallel multi-wavelength photometry. It is a python based package that uses PyTorch to quickly and efficiently perform analysis tasks. Written by [Connor Stone](https://connorjstone.com/) for tasks such as LSB imaging, handling crowded fields, multi-band photometry, and analyzing massive data from future telescopes. AutoProf is flexible and fast for any astronomical image modelling task. While it uses PyTorch (originally developed for Machine Learning) it is NOT a machine learning based tool.

## Installation

AutoProf can be installed with pip:

```
pip install autoprof
```

However, for AutoProf to run you will need to install pytorch as well. Installing pytorch is very user specific, though also not very hard. Follow the instructions on the [pytorch website](https://pytorch.org/) to install a version for your system.

Also note that AutoProf is only available for python3.

See [the documentation](https://connorstoneastro.github.io/AutoProf/) for more details.

## Documentation

You can find the documentation at the [GitHub Pages site connected with the AutoProf project](https://connorstoneastro.github.io/AutoProf/) which covers many of the main use cases for AutoProf. It is still in development, but lots of useful information is there. Feel free to contact the author, [Connor Stone](https://connorjstone.com/), for any questions not answered by the documentation or tutorials.

## Credit / Citation

If you use AutoProf in your research, please follow the [citation instructions here](https://connorstoneastro.github.io/AutoProf/citation.html). A new paper for the updated AutoProf code is in the works.

## Looking for the old AutoProf?

Don't worry, the old AutoProf is still available unchanged as *AutoProf-Legacy* simply [follow this link](https://github.com/ConnorStoneAstro/AutoProf-Legacy) to see the github page.
