===============
Troubleshooting
===============

Here we list issues that other users have encountered, and how to solve them!

**If you encounter a new issue please make an "Issue" on the GitHub so
 I can fix it!** `GitHub Issues
 <https://github.com/Autostronomy/AstroPhot/issues>`_

Jupyter notebook kernel died
----------------------------

If you repeatedly get a `Kernel Restarting` message on your jupyter
notebook with something like::

    The kernel for NOTEBOOK.ipynb appears to have died. It will restart automatically.

Then the issue might be how `PyTorch` and Jupyter talk to each
other. This is a known problem that can occur (for example `see here
<https://stackoverflow.com/questions/56759112/how-to-fix-the-kernel-appears-to-have-died-it-will-restart-automatically-caus>`_). The
solution that works for most people is to just re-install `PyTorch`
and possible `torchvision` as well.


My images look mirrored
-----------------------

If you load a target image into `AstroPhot` and then try to plot it
using one of the built-in plotting routines, you may notice that the
image is flipped horizontally. This is likely because your pixelscale
matrix has negative values such as::

  [[-1, 0]
   [ 0, 1]]

This is totally normal, it comes from the fact that `RA` is defined as
positive to the east (left side of a typical image). Most astrophot
plotting routines that may be affected by this have an argument to fix
it, just set: `flipx = True` in the plotting function and you should
be good to go!  Otherwise you can do it manually in matplotlib with
`ax.invert_xaxis()`.

My object/images aren't where they should be
--------------------------------------------

If you objects aren't where they should be in an image, or your images
aren't aligning properly, its possible that your coordinate system is
misspecified. Please read the :doc:`coordinates` documentation to get
some clarity on how AstroPhot handles coordinates. If that doesn't
work, contact me!
