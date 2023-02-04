===============
Getting Started
===============


Basic AutoProf code philosophy
------------------------------

in development.

Using the Tutorials
-------------------

The easiest way to get started using AutoProf is to try running the jupyter notebook tutorials. Simply make a new directory where you want to run the tutorials then run the::

  autoprof tutorials

command to download the AutoProf tutorials. If you run into difficulty with this, you can also access the tutorials directly at `this link <https://github.com/ConnorStoneAstro/AutoProf-2/tree/main/docs/tutorials>`_ to download. Once you have the tutorials, start a jupyter session and run through them. The recommended order is:

#. GettingStarted
#. GroupModels
#. ModelZoo
#. JointModels
#. CustomModels (if you're feeling bold)


Model Org Chart
---------------

As a quick reference for what kinds of models are available in AutoProf, the org chart shows you the class hierarchy where the leaf nodes at the bottom are the models that can actually be used. Following different paths through the hierarchy gives models with different properties. Just use the second line at each step in the flow chart to construct the name. For example one could follow a fairly direct path to get a ``sersic galaxy model``, or a more complex path to get a ``muker fourier warp galaxy model``. Note that the ``Base_Model`` object doesn't have an identifier, it is really meant to hide in the background while its subclasses do the work.

.. image:: https://github.com/ConnorStoneAstro/AutoProf-2/blob/main/media/AutoProfModelOrgchart.png?raw=true
   :alt: AutoProf Model Org Chart
   :width: 100 %

Detailed Documentation
----------------------

Detailed documentation can be found by navigating the ``autoprof`` link tree on the left. Currently it is not very organized, but detailed information can be found on just about every AutoProf system there. Further organization will come to make it easier to navigate. For now you can also just search the model type you are interested in, in the search bar.

