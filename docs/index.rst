.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

.. |br| raw:: html

    <div style="min-height:0.1em;"></div>

*********
AutoProf
*********

.. raw:: html

   <img src="../media/AP_logo.png";" width="495"/>

.. only:: latex

    .. image:: ../media/AP_logo.png

|br|

.. Important::
    If you use AutoProf for a project that leads to a publication,
    whether directly or as a dependency of another package, please
    include an :doc:`acknowledgment and/or citation <citation>`.

|br|

Getting Started
===============

AutoProf is a tool for astronomical image photometry using forward modelling.
Its design allows for fast startup and provides flexibility to explore new ideas and support advanced users.
It was written by `Connor Stone <https://connorjstone.com/>`_ .

This documentation is a work in progress, further updates will come.

.. toctree::
    :maxdepth: 1

    install.rst
    getting_started.rst
    contributing.rst
    citation.rst
    license.rst   

|br|

.. note::

    Like much astronomy software, AutoProf is an evolving package.
    I try to keep the API stable and consistent, however I will make
    changes to the interface if it considerably improves things
    going forward. Please contact connorstone628@gmail.com if you experience
    issues. If you would like to be notified of major changes send an email
    with the subject line "AUTOPROF MAILING LIST".

