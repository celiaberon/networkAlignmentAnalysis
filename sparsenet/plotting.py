import numpy as np
import matplotlib.pyplot as plt


def plot_rf(rf, out_dim, M, showRFs=None, figSize=5):
    if showRFs is not None:
        rf = rf.reshape(out_dim, -1)
        idxRandom = np.random.choice(range(rf.shape[0]), showRFs, replace=False)
        rf = rf[idxRandom, :]
    else:
        showRFs = out_dim
    # normalize
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(showRFs, M, M)
    # plotting
    n = int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(figSize, figSize)
    for i in range(rf.shape[0]):
        ax = axes[i // n][i % n]
        ax.imshow(rf[i], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for j in range(rf.shape[0], n * n):
        ax = axes[j // n][j % n]
        ax.imshow(np.ones_like(rf[0]) * -1, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig
