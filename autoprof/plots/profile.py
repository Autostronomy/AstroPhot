import matplotlib.pyplot as plt


def galaxy_profile(fig, ax, model):

    xx = np.linspace(0, np.sqrt(np.sum(model.fit_window.shape**2)), 1000)
    ax.plot(xx, np.log10(model.radial_model(xx)), linewidth = 2, label = f"{model.name} profile")
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel("Radius")

    return fig, ax

def model_image(fig, ax, model):

    model.sample_model(model.model_image)

    im = ax.imshow(
        np.log10(model.model_image.data.detach().numpy()),
        extent = model.model_image.window.plt_extent,
        origin = "lower"
    )
    clb = fig.colorbar(im, ax = ax)
    clb.ax.set_title(f"log$_{10}$(flux)")

    return fig, ax

def residual_image(fig, ax, model):
    
    model.sample_model(model.model_image)

    im = ax.imshow(
        model.target[model.model_image.window].data.detach().numpy() - model.model_image.data.detach().numpy(),
        extent = model.model_image.window.plt_extent,
        origin = "lower"
    )
    clb = fig.colorbar(im, ax = ax)
    clb.ax.set_title(f"Target - {model.name} [flux]")

    return fig, ax
