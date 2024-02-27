============
Installation
============

Basic Install
-------------

Installation is very easy for most users, simply call::

  pip install astrophot

If PyTorch gives you trouble, just follow the instructions on the `pytorch website <https://pytorch.org/>`_ which will provide a command to copy into the terminal for very easy setup.

Requirements
------------

These should automatically be installed along with AstroPhot:

numpy, scipy, matplotlib, astropy, torch, requests, tqdm

If you have difficulty running AstroPhot, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

Developer Install
-----------------

If you wish to help develop AstroPhot, thank you! You can get started by forking the repository then cloning it to your device::

  git clone https://github.com/Autostronomy/AstroPhot.git

Then you can locally install the code using::

  pip install -e .

which will make the install editable so as you make changes it will update AstroPhot. Just note that you will need to re-import astrophot for changes to take effect. For further instructions about helping with AstroPhot development see :doc:`contributing`.

Issues
------

For any install issues contact connorstone628@gmail.com for help. The code has been unit tested on Linux, Windows, and Mac machines.

