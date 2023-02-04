============
Installation
============

Requirements
------------

numpy, scipy, matplotlib, astropy, torch

If you have difficulty running AutoProf, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

Note that you must manually install torch since it is very computer specific. You can find details on the `pytorch website <https://pytorch.org/>`_ which will provide a command to copy into the terminal for very easy setup.

Basic Install
-------------

Installation is very easy for most users, simply call::

  pip install autoprof

before running AutoProf you will also need to install pytorch. This is simple, but user specific. Go to the `pytorch website <https://pytorch.org/>`_ and follow the installation instructions there.

Developer Install
-----------------

If you wish to help develop AutoProf, thank you, you can get started by cloning the repository::

  git clone git@github.com:ConnorStoneAstro/AutoProf-2.git

Then you can locally install the code using::

  pip install -e .

which will make the install editable so as you make changes it will update AutoProf. Just note that you will need to re-import autoprof for changes to take effect.

As with the basic install, you will need to make sure you have installed pytorch from the `pytorch website <https://pytorch.org/>`_ .

Issues
------

For any install issues contact connorstone628@gmail.com for help. The code has been tested on Linux (mint) and Mac machines.

