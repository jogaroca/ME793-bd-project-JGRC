"""
Small collection of plotting helpers reused across the project.

At the moment this module only contains a convenience function to plot
True / Measured / Estimated (T–M–E) time series on a single axis.  The
implementation is adapted from ``Utility/plot_utility.py`` in the
ME793 class repository.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Use slightly larger default font size, as in the class utilities
font = {"size": 18}
matplotlib.rc("font", **font)


def plot_tme(t, true, measured=None, estimated=None, ax=None, label_var: str = "y"):
    """
    Plot True / Measured / Estimated signals on a single set of axes.

    Parameters
    ----------
    t
        1D array of time stamps.
    true
        1D array with the noise-free reference trajectory.
    measured
        1D array with measured data (e.g., with noise). If None, it is omitted.
    estimated
        1D array with an estimated trajectory. If None, it is omitted.
    ax
        Optional :class:`matplotlib.axes.Axes` object.  If None, a new
        figure and axes are created.
    label_var
        Base label for the legend and y-axis (e.g. ``'B'`` or ``'N'``).

    Returns
    -------
    ax
        The :class:`matplotlib.axes.Axes` instance with the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

    if measured is not None:
        ax.plot(t, measured, "*", label=label_var + " measured", markersize=0.5)

    if estimated is not None:
        ax.plot(t, estimated, label=label_var + " hat")

    if true is not None:
        ax.plot(t, true, "--", label=label_var + " true")

    ax.set_xlabel("Time")
    ax.set_ylabel(label_var)

    ax.legend()

    return ax
