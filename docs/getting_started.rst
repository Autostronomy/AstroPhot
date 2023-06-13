===============
Getting Started
===============


Basic AutoPhot code philosophy
------------------------------

in development.

Using the Tutorials
-------------------

The easiest way to get started using AutoPhot is to try running the jupyter notebook tutorials. Simply make a new directory where you want to run the tutorials then run the::

  autophot tutorials

command to download the AutoPhot tutorials. If you run into difficulty with this, you can also access the tutorials directly at `this link <https://github.com/ConnorStoneAstro/AutoPhot/tree/main/docs/tutorials>`_ to download. Once you have the tutorials, start a jupyter session and run through them. The recommended order is:

#. GettingStarted
#. GroupModels
#. FittingMethods
#. ModelZoo
#. JointModels
#. CustomModels

When downloading the tutorials, you will also get a file called ``simple_config.py``, this is an example AutoPhot config file. Configuration files are an alternate interface to the AutoPhot functionality. They are somewhat more limited in capacity, but very easy to interface with. See the guide on configuration files here: :doc:`configfile_interface` .

Model Org Chart
---------------

As a quick reference for what kinds of models are available in AutoPhot, the org chart shows you the class hierarchy where the leaf nodes at the bottom are the models that can actually be used. Following different paths through the hierarchy gives models with different properties. Just use the second line at each step in the flow chart to construct the name. For example one could follow a fairly direct path to get a ``sersic galaxy model``, or a more complex path to get a ``muker fourier warp galaxy model``. Note that the ``Component_Model`` object doesn't have an identifier, it is really meant to hide in the background while its subclasses do the work.

.. image:: https://github.com/ConnorStoneAstro/AutoPhot/blob/main/media/AutoPhotModelOrgchart.png?raw=true
   :alt: AutoPhot Model Org Chart
   :width: 100 %

Detailed Documentation
----------------------

Detailed documentation can be found by navigating the ``autophot`` link tree on the left. Currently it is not very organized, but detailed information can be found on just about every AutoPhot system there. Further organization will come to make it easier to navigate. For now you can also just search the model type you are interested in, in the search bar.

