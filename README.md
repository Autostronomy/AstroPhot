<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/AP_logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset="media/AP_logo.png">
  <img alt="AutoProf logo" src="media/AP_logo.png" width="70%">
</picture>


![unittests](https://github.com/ConnorStoneAstro/AutoProf-2/actions/workflows/testing.yaml/badge.svg?branch=main)
![docs](https://github.com/ConnorStoneAstro/AutoProf-2/actions/workflows/documentation.yaml/badge.svg?branch=main)
![pypi](https://img.shields.io/pypi/v/autoprof.svg?logo=pypi&logoColor=white&label=PyPI)
![downloads](https://img.shields.io/pypi/dm/autoprof?label=PyPI%20Downloads)

AutoProf is a python based astronomical image modelling code. It is highly flexible for a wide range of analysis tasks and uses pytorch to accelerate calculations with automatic exact derivatives and either parallel CPU code, or by taking advantage of GPUs. While pytorch was developed for Machine Learning, AutoProf hijacks its capabilities for regular Chi squared minimization and so is not a Machine Learning tool itself. Written by [Connor Stone](https://connorjstone.com/), AutoProf was developed for a number of science goals such as LSB imaging, handling crowded fields, simultaneous multi-band image modelling, and dealing with massive volumes of data from the next generation of telescopes. Even if you aren't pushing these boundaries in particular, you will likely find AutoProf fast and easy to use for any astronomical image modelling task.

## Installation

AutoProf can be pip installed with:

```
pip install autoprof
```

However, for AutoProf to run you will need to install pytorch as well. Installing pytorch is very user specific, though also not very hard. Follow the instructions on the [pytorch website](https://pytorch.org/) to install a version for your system.

Also note that AutoProf is only available for python3.

See [the documentation](https://connorstoneastro.github.io/AutoProf-2/) for more details.

## Documentation

You can find the documentation at the [GitHub Pages site connected with the AutoProf project](https://connorstoneastro.github.io/AutoProf-2/) which covers many of the main use cases for AutoProf. It is still in development, but lots of useful information is there. Feel free to contact the author, [Connor Stone](https://connorjstone.com/), for any questions not answered by the documentation or tutorials.

## Credit / Citation

If you use AutoProf in your research, please follow the [citation instructions here](https://connorstoneastro.github.io/AutoProf-2/citation.html). A new paper for the updated AutoProf code is in the works.