#!/usr/bin/env python
# coding: utf-8

# # Advanced PSF modeling
#
# Ideally we always have plenty of well separated bright, but not oversaturated, stars to use to construct a PSF model. These models are incredibly important for certain science objectives that rely on precise shape measurements and not just total light measures. Here we demonstrate some of the special capabilities AstroPhot has to handle challenging scenarios where a good PSF model is needed but there are only very faint stars, poorly placed stars, or even no stars to work with!

# In[ ]:


import astrophot as ap
import numpy as np
import torch
import matplotlib.pyplot as plt


# ## Making a PSF model
#
# Before we can optimize a PSF model, we need to make the model and get some starting parameters. If you already have a good guess at some starting parameters then you can just enter them yourself, however if you don't then AstroPhot provides another option; if you have an empirical PSF estimate (a stack of a few stars from the field), then you can have a PSF model initialize itself on the empirical PSF just like how other AstroPhot models can initialize themselves on target images. Let's see how that works!

# In[ ]:


# First make a mock empirical PSF image
np.random.seed(124)
psf = ap.utils.initialize.moffat_psf(2.0, 3.0, 101, 0.5)
variance = psf**2 / 100
psf += np.random.normal(scale=np.sqrt(variance))

psf_target = ap.PSFImage(
    data=psf,
    pixelscale=0.5,
    variance=variance,
)

# To ensure the PSF has a normalized flux of 1, we call
psf_target.normalize()

fig, ax = plt.subplots()
ap.plots.psf_image(fig, ax, psf_target)
ax.set_title("mock empirical PSF")
plt.show()


# In[ ]:


# Now we initialize on the image
psf_model = ap.Model(
    name="init psf",
    model_type="moffat psf model",
    target=psf_target,
)

psf_model.initialize()

# PSF model can be fit to it's own target for good initial values
# Note we provide the weight map (1/variance) since a PSF_Image can't store that information.
ap.fit.LM(psf_model, verbose=1).fit()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ap.plots.psf_image(fig, ax[0], psf_model)
ax[0].set_title("PSF model fit to mock empirical PSF")
ap.plots.residual_image(fig, ax[1], psf_model, normalize_residuals=True)
ax[1].set_title("residuals")
plt.show()


# That's pretty good! it doesn't need to be perfect, so this is already in the right ballpark, just based on the size of the main light concentration. For the examples below, we will just start with some simple given initial parameters, but for real analysis this is quite handy.

# ## Group PSF Model
#
# Just like group models for regular models, it is possible to make a `psf group model` to combine multiple psf models.

# In[ ]:


psf_model1 = ap.Model(
    name="psf1",
    model_type="moffat psf model",
    n=2,
    Rd=10,
    I0=20,  # essentially controls relative flux of this component
    normalize_psf=False,  # sub components shouldnt be individually normalized
    target=psf_target,
)
psf_model2 = ap.Model(
    name="psf2",
    model_type="sersic psf model",
    n=4,
    Re=5,
    Ie=1,
    normalize_psf=False,
    target=psf_target,
)
psf_group_model = ap.Model(
    name="psf group",
    model_type="psf group model",
    target=psf_target,
    models=[psf_model1, psf_model2],
    normalize_psf=True,  # group model should normalize the combined PSF
)
psf_group_model.initialize()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ap.plots.psf_image(fig, ax[0], psf_group_model)
ax[0].set_title("PSF group model with two PSF models")
ap.plots.psf_image(fig, ax[1], psf_group_model.models[0])
ax[1].set_title("PSF model component 1")
ap.plots.psf_image(fig, ax[2], psf_group_model.models[1])
ax[2].set_title("PSF model component 2")
plt.show()


# ## PSF modeling without stars
#
# Can it be done? Let's see!

# In[ ]:


# Lets make some data that we need to fit
psf_target = ap.PSFImage(
    data=np.zeros((51, 51)),
    pixelscale=1.0,
)

true_psf_model = ap.Model(
    name="true psf",
    model_type="moffat psf model",
    target=psf_target,
    n=2,
    Rd=3,
)
true_psf = true_psf_model().data

target = ap.TargetImage(
    data=torch.zeros(100, 100),
    pixelscale=1.0,
    psf=true_psf,
)

true_model = ap.Model(
    name="true model",
    model_type="sersic galaxy model",
    target=target,
    center=[50.0, 50.0],
    q=0.4,
    PA=np.pi / 3,
    n=2,
    Re=25,
    Ie=10,
    psf_convolve=True,
)

# use the true model to make some data
sample = true_model()
torch.manual_seed(61803398)
target._data = sample.data + torch.normal(torch.zeros_like(sample.data), 0.1)
target.variance = 0.01 * torch.ones_like(sample.data.T)

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ap.plots.model_image(fig, ax[0], true_model)
ap.plots.target_image(fig, ax[1], target)
ax[0].set_title("true sersic+psf model")
ax[1].set_title("mock observed data")
plt.show()


# In[ ]:


# Now we will try and fit the data using just a plain sersic

# Here we set up a sersic model for the galaxy
plain_galaxy_model = ap.Model(
    name="galaxy model",
    model_type="sersic galaxy model",
    target=target,
)

# Let AstroPhot determine its own initial parameters, so it has to start with whatever it decides automatically,
# just like a real fit.
plain_galaxy_model.initialize()

result = ap.fit.LM(plain_galaxy_model, verbose=1).fit()
print(result.message)


# In[ ]:


# The shape of the residuals here shows that there is still missing information; this is of course
# from the missing PSF convolution to blur the model. In fact, the shape of those residuals is very
# commonly seen in real observed data (ground based) when it is fit without accounting for PSF blurring.
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ap.plots.model_image(fig, ax[0], plain_galaxy_model)
ap.plots.residual_image(fig, ax[1], plain_galaxy_model)
ax[0].set_title("fitted sersic only model")
ax[1].set_title("residuals")
plt.show()


# In[ ]:


# Now we will try and fit the data with a sersic model and a "live" psf

# Here we create a target psf model which will determine the specs of our live psf model
psf_target = ap.PSFImage(
    data=np.zeros((51, 51)),
    pixelscale=target.pixelscale,
)

live_psf_model = ap.Model(
    name="psf",
    model_type="moffat psf model",
    target=psf_target,
    n=1.0,  # True value is 2.
    Rd=2.0,  # True value is 3.
)

# Here we set up a sersic model for the galaxy
live_galaxy_model = ap.Model(
    name="galaxy model",
    model_type="sersic galaxy model",
    target=target,
    psf_convolve=True,
    psf=live_psf_model,  # Here we bind the PSF model to the galaxy model, this will add the psf_model parameters to the galaxy_model
)
live_galaxy_model.initialize()

result = ap.fit.LM(live_galaxy_model, verbose=3).fit()


# In[ ]:


print(
    f"fitted n for moffat PSF: {live_psf_model.n.value.item():.6f} +- {live_psf_model.n.uncertainty.item():.6f} we were hoping to get 2!"
)
print(
    f"fitted Rd for moffat PSF: {live_psf_model.Rd.value.item():.6f} +- {live_psf_model.Rd.uncertainty.item():.6f} we were hoping to get 3!"
)
fig, ax = ap.plots.covariance_matrix(
    result.covariance_matrix.detach().cpu().numpy(),
    live_galaxy_model.build_params_array().detach().cpu().numpy(),
    live_galaxy_model.build_params_array_names(),
)
plt.show()


# This is truly remarkable! With no stars available we were still able to extract an accurate PSF from the image! To be fair, this example is essentially perfect for this kind of fitting and we knew the true model types (sersic and moffat) from the start. Still, this is a powerful capability in certain scenarios. For many applications (e.g. weak lensing) it is essential to get the absolute best PSF model possible. Here we have shown that not only stars, but galaxies in the field can be useful tools for measuring the PSF!

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ap.plots.model_image(fig, ax[0], live_galaxy_model)
ap.plots.residual_image(fig, ax[1], live_galaxy_model)
ax[0].set_title("fitted sersic + psf model")
ax[1].set_title("residuals")
plt.show()


# There are regions of parameter space that are degenerate and so even in this idealized scenario the PSF model can get stuck. If you rerun the notebook with different random number seeds for pytorch you may find some where the optimizer "fails by immobility" this is when it gets stuck in the parameter space and can't find any way to improve the likelihood. In fact most of these "fail" fits do return really good values for the PSF model, so keep in mind that the "fail" flag only means the possibility of a truly failed fit. Unfortunately, detecting convergence is hard.

# ## PSF fitting for faint stars
#
# Sometimes there are stars available, but they are faint and it is hard to see how a reliable fit could be obtained. We have already seen how faint stars next to galaxies are still viable for PSF fitting. Now we will consider the case of isolated but faint stars. The trick here is that we have a second high resolution image, perhaps in a different band. To perform this fitting we will link up the two bands using joint modelling to constrain the star centers, this will constrain some of the parameters making it easier to fit a PSF model.

# In[ ]:


# Coming soon


# ## PSF fitting for saturated stars
#
# A saturated star is a bright star, and it's just begging to be used for modelling a PSF. There's just one catch, the highest signal to noise region is completely messed up and can't be used! Traditionally these stars are either ignored, or a two stage fit is performed to get an "inner psf" and an "outer psf" which are then merged. Why not fit the inner and outer PSFs all at once! This can be done with AstroPhot using parameter constraints and masking.

# In[ ]:


# Coming soon
