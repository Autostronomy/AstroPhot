<img src="docs/_static/AP_logo.png" alt="AutoProf" width="300"/>

AutoProf is a python based astronomical image modelling code. It is highly flexible for a wide range of analysis tasks and uses pytorch to accelerate calculations with automatic exact derivatives and either parallel CPU code, or by taking advantage of GPUs. While pytorch was developped for Machine Learning, AutoProf hijacks its capabilities for regular Chi squared minimization and so is not a Machine Learning tool itself. Written by [Connor Stone](https://connorjstone.com/), AutoProf was developped for a number of science goals such as LSB imaging, handling crowded fields, and dealing with massive volumes of data from the next generation of telescopes. Even if you aren't pushing these boundaries in particular, you will likely find AutoProf fast and easy to use for any astronomical image modelling task.

## Installation

AutoProf can be pip installed with:

```
pip install autoprof
```

However, for AutoProf to run you will need to install pytorch as well. Installing pytorch is very user specific, though also not very hard. Follow the instructions on the [pytorch website](https://pytorch.org/) to install a version for your system.

Also note that AutoProf is only available for python3.

## Getting started

The easiest way to get started using AutoProf is to try running the jupyter notebook tutorials. Simply make a new directory where you want to run the tutorials then run the:

```
autoprof tutorials
```

command to download the AutoProf tutorials. If you run into difficulty with this, you can also access the tutorials directly at [this link](https://github.com/ConnorStoneAstro/AutoProf-2/tree/main/docs/tutorials) to download. Once you have the tutorials, start a jupyter session and run through them. The recommended order is:

1. GettingStarted
1. CombinedModels
1. ModelZoo

## Documentation

Further documentation is in development. In the meantime you can contact the author [Connor Stone](https://connorjstone.com/) for any questions not answered by the tutorials.

## Credit / Citation

If you use AutoProf in your research, please credit the author by citing: [ADS Bibliographic Record](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1870S/abstract). A new paper for the updated AutoProf code is in the works.