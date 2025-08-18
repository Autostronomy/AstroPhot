#!/usr/bin/env python
# coding: utf-8

# # Aligning Images
#
# In AstroPhot, the image WCS is part of the model and so can be optimized alongside other model parameters. Here we will demonstrate a basic example of image alignment, but the sky is the limit, you can perform highly detailed image alignment with AstroPhot!

# In[ ]:


import astrophot as ap
import matplotlib.pyplot as plt
import numpy as np
import torch
import socket

socket.setdefaulttimeout(120)


# ## Relative shift
#
# Often the WCS solution is already really good, we just need a local shift in x and/or y to get things just right. Lets start by optimizing a translation in the WCS that improves the fit for our models!

# In[ ]:


target_r = ap.TargetImage(
    filename="https://www.legacysurvey.org/viewer/fits-cutout?ra=329.2715&dec=13.6483&size=150&layer=ls-dr9&pixscale=0.262&bands=r",
    name="target_r",
    variance="auto",
)
target_g = ap.TargetImage(
    filename="https://www.legacysurvey.org/viewer/fits-cutout?ra=329.2715&dec=13.6483&size=150&layer=ls-dr9&pixscale=0.262&bands=g",
    name="target_g",
    variance="auto",
)

# Uh-oh! our images are misaligned by 1 pixel, this will cause problems!
target_g.crpix = target_g.crpix + 1

fig, axarr = plt.subplots(1, 2, figsize=(15, 7))
ap.plots.target_image(fig, axarr[0], target_r)
axarr[0].set_title("Target Image (r-band)")
ap.plots.target_image(fig, axarr[1], target_g)
axarr[1].set_title("Target Image (g-band)")
plt.show()


# In[ ]:


# fmt: off
# r-band model
psfr = ap.Model(name="psfr", model_type="moffat psf model", n=2, Rd=1.0, target=target_r.psf_image(data=np.zeros((51, 51))))
star1r = ap.Model(name="star1-r", model_type="point model", window=[0, 60, 80, 135], center=[12, 9], psf=psfr, target=target_r)
star2r = ap.Model(name="star2-r", model_type="point model", window=[40, 90, 20, 70], center=[3, -7], psf=psfr, target=target_r)
star3r = ap.Model(name="star3-r", model_type="point model", window=[109, 150, 40, 90], center=[-15, -3], psf=psfr, target=target_r)
modelr = ap.Model(name="model-r", model_type="group model", models=[star1r, star2r, star3r], target=target_r)

# g-band model
psfg = ap.Model(name="psfg", model_type="moffat psf model", n=2, Rd=1.0, target=target_g.psf_image(data=np.zeros((51, 51))))
star1g = ap.Model(name="star1-g", model_type="point model", window=[0, 60, 80, 135], center=star1r.center, psf=psfg, target=target_g)
star2g = ap.Model(name="star2-g", model_type="point model", window=[40, 90, 20, 70], center=star2r.center, psf=psfg, target=target_g)
star3g = ap.Model(name="star3-g", model_type="point model", window=[109, 150, 40, 90], center=star3r.center, psf=psfg, target=target_g)
modelg = ap.Model(name="model-g", model_type="group model", models=[star1g, star2g, star3g], target=target_g)

# total model
target_full = ap.TargetImageList([target_r, target_g])
model = ap.Model(name="model", model_type="group model", models=[modelr, modelg], target=target_full)

# fmt: on
fig, axarr = plt.subplots(1, 2, figsize=(15, 7))
ap.plots.target_image(fig, axarr, target_full)
axarr[0].set_title("Target Image (r-band)")
axarr[1].set_title("Target Image (g-band)")
ap.plots.model_window(fig, axarr[0], modelr)
ap.plots.model_window(fig, axarr[1], modelg)
plt.show()


# In[ ]:


model.initialize()
res = ap.fit.LM(model, verbose=1).fit()
fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
ap.plots.model_image(fig, axarr[0], model)
axarr[0, 0].set_title("Model Image (r-band)")
axarr[0, 1].set_title("Model Image (g-band)")
ap.plots.residual_image(fig, axarr[1], model)
axarr[1, 0].set_title("Residual Image (r-band)")
axarr[1, 1].set_title("Residual Image (g-band)")
plt.show()


# Here we see a clear signal of an image misalignment, in the g-band all of the residuals have a dipole in the same direction! Lets free up the position of the g-band image and optimize a shift. This only requires a single line of code!

# In[ ]:


target_g.crtan.to_dynamic()


# Now we can optimize the model again, notice how it now has two more parameters. These are the x,y position of the image in the tangent plane. See the AstroPhot coordinate description on the website for more details on why this works.

# In[ ]:


res = ap.fit.LM(model, verbose=1).fit()
fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
ap.plots.model_image(fig, axarr[0], model)
axarr[0, 0].set_title("Model Image (r-band)")
axarr[0, 1].set_title("Model Image (g-band)")
ap.plots.residual_image(fig, axarr[1], model)
axarr[1, 0].set_title("Residual Image (r-band)")
axarr[1, 1].set_title("Residual Image (g-band)")
plt.show()


# Yay! no more dipole. The fits aren't the best, clearly these objects aren't super well described by a single moffat model. But the main goal today was to show that we could align the images very easily. Note, its probably best to start with a reasonably good WCS from the outset, and this two stage approach where we optimize the models and then optimize the models plus a shift might be more stable than just fitting everything at once from the outset. Often for more complex models it is best to start with a simpler model and fit each time you introduce more complexity.

# ## Shift and rotation
#
# Lets say we really don't trust our WCS, we think something has gone wrong and we want freedom to fully shift and rotate the relative positions of the images relative to each other. How can we do this?

# In[ ]:


def rotate(phi):
    """Create a 2D rotation matrix for a given angle in radians."""
    return torch.stack(
        [
            torch.stack([torch.cos(phi), -torch.sin(phi)]),
            torch.stack([torch.sin(phi), torch.cos(phi)]),
        ]
    )


# Uh-oh! Our image is misaligned by some small angle
target_g.CD = target_g.CD.value @ rotate(torch.tensor(np.pi / 32, dtype=torch.float64))
# Uh-oh! our alignment from before has been erased
target_g.crtan.value = (0, 0)


# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
ap.plots.model_image(fig, axarr[0], model)
axarr[0, 0].set_title("Model Image (r-band)")
axarr[0, 1].set_title("Model Image (g-band)")
ap.plots.residual_image(fig, axarr[1], model)
axarr[1, 0].set_title("Residual Image (r-band)")
axarr[1, 1].set_title("Residual Image (g-band)")
plt.show()


# Notice that there is not a universal dipole like in the shift example. Most of the offset is caused by the rotation in this example.

# In[ ]:


# this will control the relative rotation of the g-band image
phi = ap.Param(name="phi", dynamic_value=0.0, dtype=torch.float64)

# Set the target_g CD matrix to be a function of the rotation angle
# The CD matrix can encode rotation, skew, and rectangular pixels. We
# are only interested in the rotation here.
init_CD = target_g.CD.value.clone()
target_g.CD = lambda p: init_CD @ rotate(p.phi.value)
target_g.CD.link(phi)

# also optimize the shift of the g-band image
target_g.crtan.to_dynamic()


# In[ ]:


res = ap.fit.LM(model, verbose=1).fit()
fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
ap.plots.model_image(fig, axarr[0], model)
axarr[0, 0].set_title("Model Image (r-band)")
axarr[0, 1].set_title("Model Image (g-band)")
ap.plots.residual_image(fig, axarr[1], model)
axarr[1, 0].set_title("Residual Image (r-band)")
axarr[1, 1].set_title("Residual Image (g-band)")
plt.show()


# In[ ]:
