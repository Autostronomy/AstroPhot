from flow import Process
from autoprof.utils.visuals import autocmap
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import os
import numpy as np

class Plot_Model(Process):
    """
    Plots the current model image.
    """

    def action(self, state):
        autocmap.set_under("k", alpha=0)

        plt.imshow(
            state.data.model_image.data,
            origin="lower",
            cmap=autocmap,
            norm=ImageNormalize(stretch=LogStretch(), clip=False),
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                state.options.plot_path,
                f"modelimage_{state.options.name}_{state.models.iteration:04d}.jpg",
            ),
            dpi=state.options["ap_plotdpi"] if "ap_plotdpi" in state.options else 300,
        )
        plt.close()

        residual = (state.data.target - state.data.model_image).data
        plt.imshow(
            residual,
            origin="lower",
            norm=ImageNormalize(stretch=HistEqStretch(residual), clip = False),
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                state.options.plot_path,
                f"modelresidualimage_{state.options.name}_{state.models.iteration:04d}.jpg",
            ),
            dpi=state.options["ap_plotdpi"] if "ap_plotdpi" in state.options else 300,
        )
        plt.close()

        
        return state


class Plot_Loss_History(Process):
    """
    Plot the loss history for all the models to identify outliers.
    """

    def action(self, state):

        for model in state.models:
            plt.plot(list(reversed(range(len(model.loss_history)))), np.log10(model.loss_history / model.loss_history[-1]), label = model.name)
        plt.legend()
        plt.savefig(
            os.path.join(
                state.options.plot_path,
                f"losshistory_{state.options.name}.jpg",
            ),
            dpi=state.options["ap_plotdpi"] if "ap_plotdpi" in state.options else 300,
        )
        plt.close()
        
        return state
