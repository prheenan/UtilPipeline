# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt


from ..UtilGeneral import PlotUtilities
from ..UtilForce.FEC import FEC_Util, FEC_Plot
from .. import Pipeline

from ..Pipeline import Step, _plot_subdir

def nm_and_pN_limits(data,f_x):
    x_range = [[min(f_x(d)), max(f_x(d))] for d in data]
    y_range = [[min(d.Force), max(d.Force)] for d in data]
    xlim = 1e9 * np.array([np.min(x_range), np.max(x_range)])
    ylim = 1e12 * np.array([np.min(y_range), np.max(y_range)])
    return xlim,ylim

def plot_single_fec(d,f_x,xlim,ylim,i,markevery=1,callback=None,**kw):
    x = f_x(d)[::markevery] * 1e9
    f = d.Force[::markevery] * 1e12
    FEC_Plot._fec_base_plot(x,f,**kw)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if callback is not None:
        callback(i,x,f,d)
    PlotUtilities.lazyLabel("Extension (nm)", "$F$ (pN)", "")

def plot_data(base_dir,step,data,markevery=1,f_x = lambda x: x.Separation,
              xlim=None,extra_name="",**kw):
    """
    :param base_dir: where the data live
    :param step:  what step we are on
    :param data: the actual data; list of TimeSepForce
    :param markevery: how often to mark the data (useful for lowering high
    res to resonable size)
    :return: nothing, plots the data..
    """
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    name_func = FEC_Util.fec_name_func
    xlim_tmp , ylim = nm_and_pN_limits(data,f_x)
    if (xlim is not None):
        xlim = xlim
    else:
        xlim = xlim_tmp
    for i,d in enumerate(data):
        f = PlotUtilities.figure()
        plot_single_fec(d, f_x, xlim, ylim,markevery=markevery,i=i,**kw)
        out_name =   plot_subdir + name_func(0, d) +  extra_name + ".png"
        PlotUtilities.savefig(f,out_name)



def heatmap_ensemble_plot(data,out_name,xlim=None,kw_map=dict(),f_x=None,
                          xlabel="Extension (nm)"):
    """
    makes a heatmap of the ensemble, with the actual data beneath

    :param data: list of FECs
    :param out_name: what to save this as
    :return: na
    """
    if f_x is None:
        f_x = lambda x: x.Separation
    fig = PlotUtilities.figure(figsize=(3, 5))
    xlim_tmp , ylim = nm_and_pN_limits(data,f_x)
    if xlim is None:
        xlim = xlim_tmp
    ax = plt.subplot(2, 1, 1)
    FEC_Plot.heat_map_fec(data, num_bins=(200, 100),x_func=f_x,
                          use_colorbar=False,separation_max=xlim[1],**kw_map)
    for spine_name in ["bottom", "top"]:
        PlotUtilities.color_axis_ticks(color='w', spine_name=spine_name,
                                       axis_name="x", ax=ax)
    PlotUtilities.xlabel("")
    PlotUtilities.title("")
    PlotUtilities.no_x_label(ax)
    plt.xlim(xlim)
    plt.subplot(2, 1, 2)
    for d in data:
        x, f = f_x(d) * 1e9, d.Force * 1e12
        FEC_Plot._fec_base_plot(x, f, style_data=dict(color=None, alpha=0.3,
                                                      linewidth=0.5))
    PlotUtilities.lazyLabel(xlabel, "Force (pN)", "")
    plt.xlim(xlim)
    PlotUtilities.savefig(fig, out_name)