import numpy as np
import matplotlib.pyplot as plt


def plot2d(X, Y, nclasses, title):
    """
    X: [N, D]
    Y: [N]
    nclasses: number of classes
    """
    cmap = plt.get_cmap('tab10')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, label=Y, s=10)
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(nclasses))
    cbar.set_label("Class")
    
    plt.xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    plt.ylim([np.min(X[:, 1]), np.max(X[:, 1])])

    plt.xlabel('dim 0')
    plt.ylabel('dim 1')
    plt.title(title)

    # Display the plot
    plt.show()

def plot2d_hmap(d0, d1, hmap, title):
    """
    d0: [grain, grain]
    d1: [grain, grain]
    hmap: [grain, grain]
    """
    plt.pcolormesh(d0, d1, hmap, cmap='coolwarm')
    plt.colorbar()

    plt.xlabel('dim 0')
    plt.ylabel('dim 1')
    plt.title(title)
    plt.show()