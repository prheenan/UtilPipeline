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

def nm_and_pN_limits(data,f_x,f_y=None,x_convert=1e9,y_convert=1e12):
    """
    :param data: list of FECs
    :param f_x: function to get x values
    :return: tuple of xlimits,ylimits (nano units, and pN, respsecitvely)
    so that all the data will be shown.
    """
    if f_y is None:
        f_y = lambda o_: o_.Force
    x_range = [[min(f_x(d)), max(f_x(d))] for d in data]
    y_range = [[min(f_y(d)), max(f_y(d))] for d in data]
    xlim = x_convert * np.array([np.min(x_range), np.max(x_range)])
    ylim = y_convert * np.array([np.min(y_range), np.max(y_range)])
    return xlim,ylim

def plot_single_fec(d,f_x,xlim,ylim,i,markevery=1,callback=None,
                    xlabel="Extension (nm)",x_convert=1e9,y_convert=1e12,
                    f_y=None,**kw):
    """
    :param d: data to plot
    :param f_x: returns x, assumed in nano units
    :param xlim:  limits of x to use
    :param ylim:  limits of y to use
    :param i: fec index (only used for callback0
    :param markevery: see plt.plot
    :param callback: function called after plotting. takes in:
    (i,f_x(d),force in pN,d) and does whatever it wants
    :param xlabel: for the x label
    :param kw: passed to FEC_Plot._fec_base_plot
    :return:
    """
    if f_y is None:
        f_y = lambda _x: _x.Force
    x = f_x(d)[::markevery] * x_convert
    f = f_y(d)[::markevery] * y_convert
    FEC_Plot._fec_base_plot(x,f,**kw)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if callback is not None:
        callback(i,x,f,d)
    PlotUtilities.lazyLabel(xlabel, "$F$ (pN)", "")

def plot_data(base_dir,step,data,markevery=1,f_x = lambda x: x.Separation,
              xlim=None,ylim=None,extra_name="",extra_before="",dpi=200,**kw):
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
    xlim_tmp , ylim_tmp = nm_and_pN_limits(data,f_x)
    if xlim is None:
        xlim = xlim_tmp
    if ylim is None:
        ylim = ylim_tmp
    for i,d in enumerate(data):
        f = PlotUtilities.figure(dpi=dpi,figsize=(3.33,3.33))
        plot_single_fec(d, f_x, xlim, ylim,markevery=markevery,i=i,**kw)
        out_name =   plot_subdir + extra_before + name_func(0, d) + \
                     extra_name + ".png"
        PlotUtilities.savefig(f,out_name)



def heatmap_ensemble_plot(data,out_name,xlim=None,ylim=None,kw_map=dict(),
                          f_x=None,f_y=None,xlabel="Extension (nm)",dpi=200,
                          ylabel="Force (pN)",kw_singles=dict()):
    """
    makes a heatmap of the ensemble, with the actual data beneath

    :param data: list of FECs
    :param out_name: what to save this as
    :return: na
    """
    if f_x is None:
        f_x = lambda x: x.Separation
    if f_y is None:
        f_y = lambda _y: _y.Force
    fig = PlotUtilities.figure(figsize=(3, 5),dpi=dpi)
    if 'x_convert' in kw_singles and 'y_convert' in kw_singles:
        limits_kw = dict(x_convert=kw_singles['x_convert'],
                         y_convert=kw_singles['y_convert'])
    else:
        limits_kw = dict()
    xlim_tmp , ylim_tmp = nm_and_pN_limits(data,f_x,f_y=f_y,**limits_kw)
    if xlim is None:
        xlim = xlim_tmp
    if ylim is None:
        ylim = ylim_tmp
    range_x = xlim[1] - xlim[0]
    range_y = ylim[1] - ylim[0]
    n_bins_x = np.ceil(range_x)
    ratio_y_x = range_x/range_y
    n_bins_y = ratio_y_x * np.ceil(range_y)
    ax = plt.subplot(2, 1, 1)
    FEC_Plot.heat_map_fec(data, num_bins=(n_bins_x, n_bins_y),x_func=f_x,
                          y_func=f_y,use_colorbar=False,
                          separation_max=xlim[1],force_max=ylim[1],**kw_map)
    for spine_name in ["bottom", "top"]:
        PlotUtilities.color_axis_ticks(color='w', spine_name=spine_name,
                                       axis_name="x", ax=ax)
    PlotUtilities.ylabel(ylabel)
    PlotUtilities.xlabel("")
    PlotUtilities.title("")
    PlotUtilities.no_x_label(ax)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(2, 1, 2)
    kw_common = dict( style_data=dict(color=None, alpha=0.3,linewidth=0.5),
                      **kw_singles)
    for i,d in enumerate(data):
        plot_single_fec(d, f_x, xlim, ylim,i,f_y=f_y,**kw_common)
    PlotUtilities.lazyLabel(xlabel,ylabel, "")
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.savefig(fig, out_name)

def _output_heatmap(data,base,step,extra_before,name,**kw):
    out_name = Pipeline._plot_subdir(base, enum=step) + extra_before + \
               "Heat_" + name + ".png"
    heatmap_ensemble_plot(data,out_name,**kw)

def _heatmap_subplots(data_retr,base,step,extra_before,**kw_heat):
    """
    :param data_retr: list to plot
    :param base: where to save
    :param step:  See Pipeline (e.g. Step.READ)
    :param extra_before: what to put before the save name
    :param kw_heat: passed to _output_heatmap
    :return: nothing
    """
    # make the individual heatmaps
    f_x_name = _f_x_name_def()
    kw_heat_common = dict(base=base, step=step, extra_before=extra_before,
                          **kw_heat)
    for f_x, name, xlabel in f_x_name:
        _output_heatmap(data=data_retr, f_x=f_x, name=name, xlabel=xlabel,
                        **kw_heat_common)
    # make a special q VS z plot (sometime useful for energy landscapes)
    kw_q_z = dict(**kw_heat_common)
    # change the x and y limits to be the same, if we specified.
    f_sep = lambda _x: _x.ZSnsr
    f_z = lambda _x: _x.Separation
    # note that the 'y' here is the x the use specifies (separation limi)
    xlim_z, _ = nm_and_pN_limits(data_retr, f_x=f_z)
    if 'xlim' not in kw_q_z:
        xlim_sep, _ = nm_and_pN_limits(data_retr, f_x=f_sep)
        kw_q_z['ylim'] = xlim_sep
    else:
        xlim_tmp = kw_q_z['xlim']
        kw_q_z['ylim'] = xlim_tmp
        xlim_z[0] = max(xlim_z[0], xlim_tmp[0])
        xlim_z[1] = min(xlim_z[1], xlim_tmp[1])
    # always set the x limits to Z...
    kw_q_z['xlim'] = xlim_z
    # make sure to convert everything to nm
    kw_q_z_conversion = dict(x_convert=1e9, y_convert=1e9)
    kw_q_z['kw_singles'] = kw_q_z_conversion
    convert_x = lambda tmp: tmp * 1e9
    conversion = dict(ConvertX=convert_x, ConvertY=convert_x)
    kw_q_z['kw_map'] = dict(ConversionOpts=conversion)
    _output_heatmap(data=data_retr, f_x=f_z, f_y=f_sep, name="_q_vs_z",
                    xlabel="ZSnsr (nm)", ylabel="Separation (nm)", **kw_q_z)

def _f_x_name_def():
    """
    :return: a list like <function to get x, name for saving, xlabel>
    """
    to_ret = [ [lambda _x: _x.Separation,"_Sep","Extension (nm)"],
               [lambda _x: _x.ZSnsr     ,"_Z","ZSnsr (nm)"] ]
    return to_ret

def _filter_f(data,f_filter=None):
    if f_filter is not None and f_filter > 0:
        n = int(np.ceil(data[0].Force.size * f_filter))
        # filter the data first
        data_retr = [FEC_Util.GetFilteredForce(d,n) for d in data]
    else:
        data_retr = data
    return data_retr

def _debug_plot_data(data_retr,base,step,extra_before="",cb=None,f_filter=None,
                     kw_heat=dict(),kw_data=dict(),plot_each=True):
    """
    :param data_retr: list of timesepforce object to use
    :param base: base directory
    :param step: step to use
    :param extra_before: add to the plotting string...
    :param cb: callback, see plot_data. Only used on separation plot.
    :param kw_heat: dictionary passed to heatmap_ensemble_plot
    :param kw_data: dictionary passed to plot_data
    :param plot_each: if true, plots all individual data
    :return: nothing
    """
    data_retr = _filter_f(data_retr, f_filter=f_filter)
    # make a plot of the individual data points; make sure they are filtered.
    f_x_name = _f_x_name_def()
    _heatmap_subplots(data_retr, base, step, extra_before, **kw_heat)
    if not plot_each:
        return
    # make a 'q vs z' plot (~ q vs time, but important for landsacpe stuff)
    for i,(f_x,name,xlabel) in enumerate(f_x_name):
        callback_tmp = cb if i==0 else None
        extra_before_tmp = extra_before + name
        plot_data(base_dir=base, step=step, data=data_retr,
                  callback=callback_tmp,extra_before=extra_before_tmp,
                  xlabel=xlabel,f_x=f_x,**kw_data)

def gallery_plot(fecs_refold,out_path,f_x=lambda _x: _x.ZSnsr,x_convert=1e9,
                 xlabel="ZSnsr (nm)",max_gallery=None):
    """
    :param fecs_refold: to plot
    :param out_path:  where the gallery plot should go
    :return:  nothing
    """
    if max_gallery is not None:
        fecs_refold = fecs_refold[:max_gallery]
    kw_savefig_ind=dict(subplots_adjust=dict(hspace=0.02,wspace=0.02))
    n_fecs = len(fecs_refold)
    inch_per_fec = 1.2
    n_side = int(np.ceil(np.sqrt(n_fecs)))
    size = n_side * inch_per_fec * np.array([1,1])
    fig = PlotUtilities.figure(size)
    xlim, ylim = nm_and_pN_limits(fecs_refold,f_x=f_x,x_convert=x_convert)
    last_row_first_element = n_side * (n_side-1)
    for i,r in enumerate(fecs_refold):
        ax = plt.subplot(n_side,n_side,(i+1))
        plot_single_fec(r, f_x=f_x, xlim=xlim, ylim=ylim, i=i,xlabel=xlabel)
        if (i != 0):
            PlotUtilities.ylabel("")
            PlotUtilities.no_y_label(ax=ax)
        if (i != last_row_first_element):
            PlotUtilities.xlabel("")
            PlotUtilities.no_x_label(ax=ax)
    PlotUtilities.savefig(fig,out_path,**kw_savefig_ind)


def _exhaustive_debug_plot(objs,base,step,f_filter=0.01,extra_before="",
                           max_gallery=25,**kw_common):
    """
    :param objs: list of fecs; will output filtered and unfiltered information
    :param base:  base place to save
    :param step:  which step we are using
    :param f_filter: fraction to filer
    :param max_gallery: maximum number to plot on the gallery.
    :param kw_common: passded to _debug_plot_data
    :return:
    """
    _debug_plot_data(data_retr=objs,base=base,step=step,
                     f_filter=f_filter,extra_before="filtered" + extra_before,
                     **kw_common)
    _debug_plot_data(data_retr=objs,base=base,step=step,**kw_common)
    plot_subdir = Pipeline._plot_subdir(base, step)
    for  f_x,name,label in _f_x_name_def():
        out_path = plot_subdir + extra_before + "Gallery_{:s}.png".format(name)
        gallery_plot(objs,out_path=out_path,xlabel=name,f_x=f_x,x_convert=1e9,
                     max_gallery=max_gallery)