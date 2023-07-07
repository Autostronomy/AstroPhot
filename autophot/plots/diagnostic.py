import numpy as np
import torch

from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import iqr
from scipy.stats import norm

__all__ = ("covariance_matrix", )

def covariance_matrix(covariance_matrix, mean, labels = None, figsize = (10,10), reference_values = None, ellipse_colors='g', showticks = True, **kwargs):
    num_params = covariance_matrix.shape[0]
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize)
    plt.subplots_adjust(wspace=0., hspace=0.)

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            if i == j:
                x = np.linspace(mean[i] - 3 * np.sqrt(covariance_matrix[i, i]), mean[i] + 3 * np.sqrt(covariance_matrix[i, i]), 100)
                y = norm.pdf(x, mean[i], np.sqrt(covariance_matrix[i, i]))
                ax.plot(x, y, color='g')
                ax.set_xlim(mean[i] - 3 * np.sqrt(covariance_matrix[i, i]), mean[i] + 3 * np.sqrt(covariance_matrix[i, i]))
                if reference_values is not None:
                    ax.axvline(reference_values[i], color='red', linestyle='-', lw=1)
            elif j < i:
                cov = covariance_matrix[np.ix_([j, i], [j, i])]
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
                for k in [1, 2]:
                    ellipse = Ellipse(xy=(mean[j], mean[i]),
                                      width=lambda_[0] * k * 2,
                                      height=lambda_[1] * k * 2,
                                      angle=angle,
                                      edgecolor=ellipse_colors,
                                      facecolor='none')
                    ax.add_artist(ellipse)

                # Set axis limits
                margin = 3
                ax.set_xlim(mean[j] - margin * np.sqrt(covariance_matrix[j, j]), mean[j] + margin * np.sqrt(covariance_matrix[j, j]))
                ax.set_ylim(mean[i] - margin * np.sqrt(covariance_matrix[i, i]), mean[i] + margin * np.sqrt(covariance_matrix[i, i]))
                
                if reference_values is not None:
                    ax.axvline(reference_values[j], color='red', linestyle='-', lw=1)
                    ax.axhline(reference_values[i], color='red', linestyle='-', lw=1)
                
            if j > i:
                ax.axis('off')

            if i < num_params - 1:
                ax.set_xticklabels([])
            else:
                if labels is not None:
                    ax.set_xlabel(labels[j])
            if not showticks:
                ax.yaxis.set_major_locator(plt.NullLocator())

            if j > 0:
                ax.set_yticklabels([])
            else:
                if labels is not None:
                    ax.set_ylabel(labels[i])
            if not showticks:
                ax.xaxis.set_major_locator(plt.NullLocator())

    
    return fig, ax

if __name__ == "__main__":

    fig, ax = covariance_matrix(np.array([[4,-2], [-2,4]]), np.array([0,0]))
    plt.show()
