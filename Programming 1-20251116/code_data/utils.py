import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_2d_decisionboundary(model, X, y, grid_resolution=0.1, color_data={'colors': ['r', 'b', 'g', 'y'], 'c_light': ['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFFFAA'],
    'c_bold': ['#FF0000', '#0000FF', '#00FF00', '#FFFF00']}, xticks=[], yticks=[], title="", show_legend=True, savefig_path=None):
    """
    Plot the decision boundary of a classifier for a 2d data set.
    Parameters
    ----------
    model: object
        Model/Classifier must implement a `predict` method.
    X : numpy array
        1d or 2d data set. Data points are expected to be stored row-wise.
    y : numpy array
        1d array containing the labels (integers).
    """
    
    h = grid_resolution
    cmap_light = ListedColormap(color_data['c_light'][:len(np.unique(y))])
    cmap_bold = ListedColormap(color_data['c_bold'][:len(np.unique(y))])

    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    # Compute the prediction for each grid cell

    # Plot
    plt.figure()
    for c in color_data['colors']:
        plt.scatter([], [], c=c)

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='nearest')
    plt.scatter(X[:, 0], X[:, 1], c=list(map(lambda i: color_data['colors'][i], y)), edgecolor='k', s=20)
    
    if 'prototypes0' in model.__dict__:
        plt.scatter(model.prototypes0[:, 0], model.prototypes0[:, 1], c=cmap_bold.colors[0], marker='*', edgecolor='k', s=250)
        plt.scatter(model.prototypes1[:, 0], model.prototypes1[:, 1], c=cmap_bold.colors[1], marker='*', edgecolor='k', s=250)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(title)

    if show_legend is True:
        plt.legend(['Class {0}'.format(i) for i in np.unique(y)])
    
    if savefig_path is None:
        plt.show()
    else:
        plt.savefig(savefig_path)

