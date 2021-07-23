import pickle
import numpy as np
import scipy
import pandas as pd
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import optimize
          
import tkinter as tk
from tkinter import simpledialog

# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import matplotlib.widgets as mwidgets
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from numba import jit, njit, float32, int32

# from PyQt4 import QtGui
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

# mpl.use('Qt5Agg')
# from PyQt5 import QtWidgets
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# class ScrollableWindow(QtWidgets.QMainWindow):
#     '''Source: https://stackoverflow.com/questions/42622146/scrollbar-on-matplotlib-showing-page
#     '''
#     def __init__(self, fig):
#         self.qapp = QtWidgets.QApplication([])

#         QtWidgets.QMainWindow.__init__(self)
#         self.widget = QtWidgets.QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QtWidgets.QVBoxLayout())
#         self.widget.layout().setContentsMargins(0,0,0,0)
#         self.widget.layout().setSpacing(0)

#         self.fig = fig
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.draw()
#         self.scroll = QtWidgets.QScrollArea(self.widget)
#         self.scroll.setWidget(self.canvas)

#         self.nav = NavigationToolbar(self.canvas, self.widget)
#         self.widget.layout().addWidget(self.nav)
#         self.widget.layout().addWidget(self.scroll)

#         self.show()
#         exit(self.qapp.exec_()) 


##############################################################################

# class ScrollableWindow(QtGui.QMainWindow):
#     '''Source: https://stackoverflow.com/questions/42622146/scrollbar-on-matplotlib-showing-page
#     '''
#     def __init__(self, fig):
#         self.qapp = QtGui.QApplication([])

#         QtGui.QMainWindow.__init__(self)
#         self.widget = QtGui.QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QtGui.QVBoxLayout())
#         self.widget.layout().setContentsMargins(0,0,0,0)
#         self.widget.layout().setSpacing(0)

#         self.fig = fig
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.draw()
#         self.scroll = QtGui.QScrollArea(self.widget)
#         self.scroll.setWidget(self.canvas)

#         self.nav = NavigationToolbar(self.canvas, self.widget)
#         self.widget.layout().addWidget(self.nav)
#         self.widget.layout().addWidget(self.scroll)

#         self.show()
#         exit(self.qapp.exec_()) 

##############################################################################

def period_for_dP_plot(periods,mode='middle'):
    '''
    Purpose:
        Return the array of periods with one less element to enable the plot
        with its period differences 
    '''
    
    if mode == 'middle':
        return (periods[1:]+periods[:-1])/2.
    
    elif mode == 'right':
        return periods[1:]
    
    elif mode == 'left':
        return periods[:-1]
    
    else:
        raise ValueError(f'`mode` is: {mode}. It has to be one of the following values: "middle", "right", "left".')

##############################################################################

@jit(nopython=True)    
def pattern_period(P0,
                   dP0,
                   Sigma,
                   nr=5,
                   nl=5):
    '''
    Purpose:
        Return a sequence of periods following the parametrization of equation
        8 in Li Gang 2019a.
    
    Args:
        P0 : float
            Position of the central peak
    
        dP0 : float
            Period spacing scale
            
        Sigma : float
            Rate of change of the period spacing
            
        nr : int
            Number of periods to the right of `P0`, including `P0`.
            
        nl : int
            Number of periods to the left of `P0`, not including `P0`.
    '''

    # Li Gang 2019a, equation 8
    ## Eq8 = lambda i: dP0 * ((1+Sigma)**i-1)/Sigma + P0
    
    indices = np.arange(-nl,nr)
    
    if Sigma==0:
        P = dP0*indices + P0
        dP = np.repeat(dP0,indices.size)
        return np.vstack((P, dP))
    else:
        P = dP0 * ( (1+Sigma)**indices-1 )/Sigma + P0
        dP = dP0 * (1+Sigma)**indices
        return np.vstack((P, dP))
    
##############################################################################

def plot_comb(interactive=False, idle=True):

    global ax_dp, ax_echelle, ax_pg, plotted_lines, comb_params

    if interactive==True:
        # Clear plot
        for line in plotted_lines['comb_pg']:
            clear_line2D(fig, line, ax_pg, redraw=False)
        for line in plotted_lines['comb_dp']:
            clear_line2D(fig, line, ax_dp, redraw=False)

    ### Compute new values
    peaks, dpeaks = pattern_period(P0=comb_params['P0'], dP0=comb_params['dP0'], Sigma=comb_params['Sigma'], nr=comb_params['nr'], nl=comb_params['nl'])
    comb_params['P']  = peaks
    comb_params['dP'] = dpeaks
    
    _P  = comb_params['P']
    _dP = comb_params['dP']
    freq_resolution = 2.5*(1./365) # units of 1/day
    dfreq = np.abs(_dP/(_P**2+_P*_dP))
    unresolved = dfreq <= freq_resolution
    
    ### Periodogram
    del plotted_lines['comb_pg'][:]
    # Plot all periods but P0
    for p,under in zip(peaks,unresolved):  
        if p != comb_params['P0']:
            color,ls = ('r','solid') if not under else ('darkviolet','dashed')
            line = ax_pg.axvline(p, color=color, ls=ls, alpha=0.3, lw=2, zorder=0)  
            plotted_lines['comb_pg'].append(line)
    # Plot P0 with another color
    line = ax_pg.axvline(comb_params['P0'], color='gold', alpha=0.9, lw=2, zorder=0)
    plotted_lines['comb_pg'].append(line)

    ### dP vs P plot
    del plotted_lines['comb_dp'][:]
    x = period_for_dP_plot(comb_params['P'], mode='middle')
    y = np.diff(comb_params['P'])                                                                                                            
    xlim = ax_dp.get_xlim()
    ylim = ax_dp.get_ylim()
    line, = ax_dp.plot(x, y, lw=1, color='r', marker='*', ls='solid', zorder=1, alpha=0.5)
    plotted_lines['comb_dp'].append(line)
    # Mark the comb dP0 in the dP plot
    if comb_params['nr'] > 1:
        ind = np.abs(comb_params['P']-comb_params['P0']).argmin()
        _ = comb_params['P'][ind:ind+2]
        x = period_for_dP_plot(_, mode='middle')
        y = np.diff(_)   
        line, = ax_dp.plot(x, y, lw=1, color='gold', marker='*', ls='None', zorder=1, alpha=0.5)
        plotted_lines['comb_dp'].append(line)
    ax_dp.set_xlim(xlim)
    ax_dp.set_ylim(ylim)
        
    ### Echelle
    update_echelle_comb()

    ### Redraw
    if interactive==True:
        if idle==True:
            fig.canvas.draw_idle()
        else:
            fig.canvas.draw()
            
##############################################################################

#### Update values

def update_comb(val):
    # Retrieve values from plot
    comb_params['P0']    = slider_P0.val
    comb_params['dP0']   = slider_dP0.val
    comb_params['Sigma'] = slider_Sigma.val
    plot_comb(interactive=True)

def read_box_P0(text):
    P0 = np.float(text)
    slider_P0.set_val(P0)
    comb_params['P0'] = P0
    plot_comb(interactive=True)
    
def read_box_dP0(text):
    dP0 = np.float(text)
    slider_dP0.set_val(dP0)
    comb_params['dP0'] = dP0
    plot_comb(interactive=True)
    
def read_box_Sigma(text):
    Sigma = np.float(text)
    slider_Sigma.set_val(Sigma)
    comb_params['Sigma'] = Sigma
    plot_comb(interactive=True)

##############################################################################

def update_echelle(val):
    
    global scatter1, scatter2, scatter3, ax_dp, ax_echelle, plotted_lines, dp
    
    # Clear plot
    scatter1.remove()
    scatter2.remove()
    scatter3.remove()
    
    colors = [choose_color(val) for val in df.selection.values]
    dp = slider_mod.val
    # Observations
    scatter1 = ax_echelle.scatter((df.period)%dp - dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter2 = ax_echelle.scatter((df.period)%dp + dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter3 = ax_echelle.scatter((df.period)%dp,      df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    
    for line in plotted_lines['echelle_vline']:
        clear_line2D(fig, line, ax_echelle, redraw=False)
    del plotted_lines['echelle_vline'][:]
    line = ax_echelle.axvline(dp, ls='dashed', color='gray', lw=2, zorder=2)
    plotted_lines['echelle_vline'].append(line)

    update_echelle_comb()
    ax_echelle.set_xlim(-dp,2.*dp)

    ax_echelle.set_xlabel(f'period mod {dp:.5f} (days)')

    for line in plotted_lines['mod_dp']:
        clear_line2D(fig, line, ax_dp, redraw=False)
    del plotted_lines['mod_dp'][:]
    line = ax_dp.axhline(dp, color='dodgerblue', lw=1, zorder=0, ls='dotted')
    plotted_lines['mod_dp'].append(line)
    
    fig.canvas.draw_idle()      

##############################################################################
    
def read_keystroke(event):
    '''Get the pressed key over the axes during plot visualization'''
    ivar['keystroke'] = event.key

##############################################################################

def swap_value(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
    else:
        raise ValueError('Input is different from 0 and 1.')

##############################################################################

def choose_color(x):
    if x == 0:
        return 'lightgrey'
    elif x == 1:
        return 'k'
    else:
        raise ValueError('Input is different from 0 and 1.')   

##############################################################################

def update_amplitude_tolerance(val):

    global scatter1, scatter2, scatter3, df, df_full, dp, plotted_lines, ax_dp, ax_echelle, ax_p

    # Bounds of the current selection
    pmin = df.query('selection==1').period.min()
    pmax = df.query('selection==1').period.max()
    bounds = (df.period >= pmin) & (df.period <= pmax)

    # amp_max = df.query('period >= @pmin and period <= @pmax').amp.max()
    amp_max = df[bounds].amp.max()
    ind = df.amp/amp_max >= val


    if np.sum((ind & bounds)) >= 3: # to avoid conflict with: `slider_mod.ax.set_xlim(y.min(),y.max())`
        df.loc[ind & bounds,'selection']  = 1
        df.loc[~ind & bounds,'selection'] = 0

    # Clear plot
    for line in plotted_lines['selection_p']:
        clear_line2D(fig, line, ax_p, redraw=False)
    for line in plotted_lines['obs_dp']:
        clear_line2D(fig, line, ax_dp, redraw=False)  

     ### Plot prewhitening dP vs P
    del plotted_lines['obs_dp'][:]
    pmin = df.query('selection==1').period.min()
    pmax = df.query('selection==1').period.max()
    x = period_for_dP_plot(df.query('period >= @pmin and period <= @pmax').period.values, mode='middle')
    y = np.diff(df.query('period >= @pmin and period <= @pmax').period.values)
    line, = ax_dp.plot(x, y, lw=1, color='lightgrey', ls='dashed', marker='.', zorder=2, picker=5)
    plotted_lines['obs_dp'].append(line)
    x = period_for_dP_plot(df.query('selection==1').period.values, mode='middle')
    y = np.diff(df.query('selection==1').period.values)
    line, = ax_dp.plot(x, y, lw=1, color='k', ls='dashed', marker='.', zorder=2, picker=5)
    plotted_lines['obs_dp'].append(line)       

    # Update the mod slider accordingly
    if y.min() != y.max():
        slider_mod.ax.set_xlim(y.min(),y.max())

    ### Plot available periods for the fit
    del plotted_lines['selection_p'][:]
    x = df.query('selection==1').period.values
    y = np.repeat(0.2, x.size)
    line, = ax_p.plot(x, y, color='k', marker=7, alpha=1, zorder=2, picker=5, ls='None')
    plotted_lines['selection_p'].append(line)

    ### Plot not-available periods for the fit
    x = df.query('selection==0').period.values
    y = np.repeat(0.2, x.size)
    line, = ax_p.plot(x, y, color='lightgrey', marker=7, alpha=1, zorder=2, picker=5, ls='None')
    plotted_lines['selection_p'].append(line)

    ### Update ecchelle
    scatter1.remove()
    scatter2.remove()
    scatter3.remove()
    colors = [choose_color(val) for val in df.selection.values]
    #dp = np.median(np.diff(df.period.values))
    xlim = ax_echelle.get_xlim()
    ylim = ax_echelle.get_ylim()
    scatter1 = ax_echelle.scatter((df.period)%dp - dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter2 = ax_echelle.scatter((df.period)%dp + dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter3 = ax_echelle.scatter((df.period)%dp,      df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    update_echelle_comb()
    ax_echelle.set_xlim(xlim)
    ax_echelle.set_ylim(ylim)


    fig.canvas.draw_idle()   
##############################################################################
        
def update_selection(event):
    
    global scatter1, scatter2, scatter3, df, df_full, dp, plotted_lines, ax_dp, ax_echelle, ax_p
    
    # Only continue if `control` is the last keystroke
    if not ivar['keystroke']=='control':
        return
    
    # Only continue after a left click on the axis ax_p or ax_echelle
    if not (event.mouseevent.inaxes==ax_p or event.mouseevent.inaxes==ax_echelle) or not event.mouseevent.button==1:
        return

    # Clear plot
    for line in plotted_lines['selection_p']:
        clear_line2D(fig, line, ax_p, redraw=False)
    for line in plotted_lines['obs_dp']:
        clear_line2D(fig, line, ax_dp, redraw=False)            

    if event.mouseevent.inaxes==ax_p:
        # Invert selection ### Make sure df has reset indexes
        i = (df.period-event.mouseevent.xdata).abs().argmin()
        df.loc[i,'selection'] = swap_value(df.loc[i,'selection']) 
    if event.mouseevent.inaxes==ax_echelle:
        # Invert selection ### Make sure df has reset indexes
        i = (df.period-event.mouseevent.ydata).abs().argmin()
        df.loc[i,'selection'] = swap_value(df.loc[i,'selection']) 

    ### Plot prewhitening dP vs P
    del plotted_lines['obs_dp'][:]
    pmin = df.query('selection==1').period.min()
    pmax = df.query('selection==1').period.max()
    x = period_for_dP_plot(df.query('period >= @pmin and period <= @pmax').period.values, mode='middle')
    y = np.diff(df.query('period >= @pmin and period <= @pmax').period.values)
    line, = ax_dp.plot(x, y, lw=1, color='lightgrey', ls='dashed', marker='.', zorder=2, picker=5)
    plotted_lines['obs_dp'].append(line)
    x = period_for_dP_plot(df.query('selection==1').period.values, mode='middle')
    y = np.diff(df.query('selection==1').period.values)
    line, = ax_dp.plot(x, y, lw=1, color='k', ls='dashed', marker='.', zorder=2, picker=5)
    plotted_lines['obs_dp'].append(line)

    # Update the mod slider accordingly
    if y.min() != y.max():
        slider_mod.ax.set_xlim(y.min(),y.max())

    ### Plot available periods for the fit
    del plotted_lines['selection_p'][:]
    x = df.query('selection==1').period.values
    y = np.repeat(0.2, x.size)
    line, = ax_p.plot(x, y, color='k', marker=7, alpha=1, zorder=2, picker=5, ls='None')
    plotted_lines['selection_p'].append(line)

    ### Plot not-available periods for the fit
    x = df.query('selection==0').period.values
    y = np.repeat(0.2, x.size)
    line, = ax_p.plot(x, y, color='lightgrey', marker=7, alpha=1, zorder=2, picker=5, ls='None')
    plotted_lines['selection_p'].append(line)

    ### Update ecchelle
    scatter1.remove()
    scatter2.remove()
    scatter3.remove()
    colors = [choose_color(val) for val in df.selection.values]
    #dp = np.median(np.diff(df.period.values))
    xlim = ax_echelle.get_xlim()
    ylim = ax_echelle.get_ylim()
    scatter1 = ax_echelle.scatter((df.period)%dp - dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter2 = ax_echelle.scatter((df.period)%dp + dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    scatter3 = ax_echelle.scatter((df.period)%dp,      df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
    update_echelle_comb()
    ax_echelle.set_xlim(xlim)
    ax_echelle.set_ylim(ylim)


    fig.canvas.draw()      

##############################################################################

def update_echelle_comb(keep_xylim=False):
    
    global scatter11, scatter22, scatter33, scatter44
    
    if keep_xylim:
        xlim = ax_echelle.get_xlim()
        ylim = ax_echelle.get_ylim()
    scatter11.remove()
    scatter22.remove()
    scatter33.remove()
    scatter44.remove()

    scatter11 = ax_echelle.scatter((comb_params['P'])%dp - dp, comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
    scatter22 = ax_echelle.scatter((comb_params['P'])%dp + dp, comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
    scatter33 = ax_echelle.scatter((comb_params['P'])%dp,      comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
    scatter44 = ax_echelle.scatter((comb_params['P0'])%dp,     comb_params['P0'], s=30, color='gold', zorder=4, alpha=0.3, marker='*')

    if keep_xylim:
        ax_echelle.set_xlim(xlim)
        ax_echelle.set_ylim(ylim)

##############################################################################
    
def read_button(event):
    
    # Only continue if the last keystroke is `control`
    if not ivar['keystroke']=='control':
        return
    
    # Periodogram
    if event.inaxes==ax_pg:
        # Add 1 line
        if event.button==1:
            if event.xdata < comb_params['P0']: # click on template comb's left
                comb_params['nl'] += 1
            else:
                comb_params['nr'] += 1
        # Remove 1 line
        if event.button==3:
            if event.xdata < comb_params['P0']: # click on template comb's left
                comb_params['nl'] -= 1
            else:
                comb_params['nr'] -= 1
        # Add/Remove 5 lines
        if event.button==2:
            if event.xdata < comb_params['P0']: # click on template comb's left
                comb_params['nl'] += 5 if event.xdata < comb_params['P'][0]  else -5
            else:
                comb_params['nr'] += 5 if event.xdata > comb_params['P'][-1] else -5

        # Ensure acceptable value
        comb_params['nr'] = max(comb_params['nr'],1)
        comb_params['nl'] = max(comb_params['nl'],0)

    # Plot dP vs P
    if event.inaxes==ax_dp:
        _P0  = event.xdata - event.ydata/2
        _dP0 = event.ydata
        # Set P0 and dP0
        if event.button==1:
            slider_P0.set_val(_P0)
            comb_params['P0']  = _P0
            slider_dP0.set_val(_dP0)
            comb_params['dP0'] = _dP0
        # Set Sigma
        if event.button==3:
            Sigma = (comb_params['dP0']-_dP0)/(comb_params['P0']-_P0) # slope formula
            slider_Sigma.set_val(Sigma)
            comb_params['Sigma'] = Sigma
                    
    # Correlation plot dP0 and Sigma
    if event.inaxes==ax_res_dP0Sigma:
        # Set dP0 and Sigma
        if event.button==1:
            slider_dP0.set_val(event.xdata)
            comb_params['dP0'] = event.xdata
            slider_Sigma.set_val(event.ydata)
            comb_params['Sigma'] = event.ydata

    # Correlation plot P0 and dP0
    if event.inaxes==ax_res_P0dP0:
        # Set P0 and dP0
        if event.button==1:
            slider_P0.set_val(event.xdata)
            comb_params['P0'] = event.xdata
            slider_dP0.set_val(event.ydata)
            comb_params['dP0'] = event.ydata

    # Correlation plot Sigma and P0
    if event.inaxes==ax_res_SigmaP0:
        # Set Sigma and P0
        if event.button==1:
            slider_Sigma.set_val(event.xdata)
            comb_params['Sigma'] = event.xdata
            slider_P0.set_val(event.ydata)
            comb_params['P0'] = event.ydata

    # Plot P0
    if event.inaxes==ax_res_P0:
        # Set P0
        if event.button==1:
            slider_P0.set_val(event.xdata)
            comb_params['P0'] = event.xdata

    # Plot dP0
    if event.inaxes==ax_res_dP0:
        # Set dP0
        if event.button==1:
            slider_dP0.set_val(event.xdata)
            comb_params['dP0'] = event.xdata

    # Plot Sigma
    if event.inaxes==ax_res_Sigma:
        # Set Sigma
        if event.button==1:
            slider_Sigma.set_val(event.xdata)
            comb_params['Sigma'] = event.xdata

    plot_comb(interactive=True) 


##############################################################################

def S(params,
      nr,
      nl,
      P_obs,
      weight,
      sigma):
    '''
    Same as _S but collects the parameters to optimize in the first argument.
    It does it to comply with the convention of scipy.optimize.minimize
    '''
    return _S(*params,nr,nl,P_obs,weight,sigma)

##############################################################################

@njit
def _S(P0,
       dP0,
       alpha,
       nr,
       nl,
       P_obs,
       weight,
       sigma):

    P_model, dP_model = pattern_period(P0,
                                       dP0,
                                       alpha,
                                       nr=nr,
                                       nl=nl)
    
    # If none, generate weight and sigma
    if weight is None:
        weight = np.repeat(1.,P_obs.size)
    if sigma is None:
        sigma = np.repeat(0.,P_obs.size)
    
    # Iterate over the terms of the sum
    S = 0
    for p_i,w_i,sigma_i in zip(P_obs,weight,sigma):
    
        i = np.abs(p_i-P_model).argmin()
        p_model  = P_model[i]
        dp_model = dP_model[i]

        S += w_i*(p_i-p_model)**2/(dp_model**2+sigma_i**2)

    return S

###################

def do_fit3(event):

    global comb_params, plotted_lines, ax_p
    
    P_obs        = df.query('selection==1').period.values
    e_P_obs      = df.query('selection==1').e_period.values
    A_obs        = df.query('selection==1').amp.values
    weights_obs  = A_obs/A_obs.max()
    weights_obs /= weights_obs.sum() # normalize the weights

    ### Minimization parameters

    x0 = [comb_params['P0'],comb_params['dP0'],comb_params['Sigma']]
    
    freq_resolution = 2.5*(1/365) # in 1/days (same as in the prewhitening)
    tolerance = 0.5
    freq_lower_bound = (1/comb_params['P0']) - tolerance*freq_resolution/2
    freq_upper_bound = (1/comb_params['P0']) + tolerance*freq_resolution/2
    
    P0_bounds    = (1/freq_upper_bound,  1/freq_lower_bound)
    dP0_bounds   = (lower_limit_dP0,     upper_limit_dP0)
    Sigma_bounds = (lower_limit_Sigma,   upper_limit_Sigma)
    
    bounds = [P0_bounds, dP0_bounds, Sigma_bounds]
    
    args = (comb_params['nr'], comb_params['nl'], P_obs, weights_obs, e_P_obs)

    # Minimization

#     **FROM SOURCE CODE**
#     if method is None:
#     # Select automatically
#     if constraints:
#         method = 'SLSQP'
#     elif bounds is not None:
#         method = 'L-BFGS-B'
#     else:
#         method = 'BFGS'

    results = minimize(S, x0, args=args, bounds=bounds)
    
    comb_params['P0']    = results.x[0]
    comb_params['dP0']   = results.x[1]
    comb_params['Sigma'] = results.x[2]
    comb_params['data']  = df.copy()
    comb_params['opt']   = results
    
#     # Uncertainties
#     # Source: https://stackoverflow.com/a/53489234/9290590
#     ftol = 2.220446049250313e-09
#     tmp_i = np.zeros(len(results.x))
#     uncertainties = []
#     for i in range(len(results.x)):
#         tmp_i[i] = 1.0
#         hess_inv_i = results.hess_inv(tmp_i)[i]
#         uncertainty_i = np.sqrt(max(1, abs(results.fun)) * ftol * hess_inv_i)
#         tmp_i[i] = 0.0
#         uncertainties.append(uncertainty_i)
        
#     comb_params['e_P0'] = uncertainties[0]
#     comb_params['e_dP0'] = uncertainties[1]
#     comb_params['e_Sigma'] = uncertainties[2]

    # Update velues
    slider_P0.set_val(results.x[0])
    slider_dP0.set_val(results.x[1])
    slider_Sigma.set_val(results.x[2])

    plot_comb(interactive=True, idle=False)
    
    ### Select the periods in the comb that agree with the data (observations) within dP/4
    
    # Find periods in template that matches observations within % of dP
    tolerance_match = 0.25
    _ = df.query('selection==1').period.values
    distances = [ np.abs(_ - p).min() for p in comb_params['P'] ]
    distances = np.array(distances) 
    imatch = distances < tolerance_match*comb_params['dP']
    comb_p_match = comb_params['P'][imatch]
    # Find corresponding observations
    obs_p_match = [ _[np.abs(_ - p).argmin()] for p in comb_p_match ]
    obs_p_match = np.unique(obs_p_match)
    # Collect
    comb_params['match'] = obs_p_match
    
    # Find mismatch
    distances = []
    normalized_distances = []
    for p_obs in comb_params['match']:
        _ = p_obs - comb_params['P']
        distance = np.abs(_).min()
        local_dp = comb_params['dP'][np.abs(_).argmin()]
        distances.append(distance)
        normalized_distances.append( distance/local_dp )
    residuals = np.sum(distances)
    mismatch = np.sum(normalized_distances)/comb_params['match'].size
    # Collect
    comb_params['residuals'] = residuals
    comb_params['mismatch'] = mismatch
    # Update boxes
    text_box_mismatch.set_val(f'{mismatch:.6f}')
    text_box_residuals.set_val(f'{residuals:.6f}')
    
    # Clear plot
    for line in plotted_lines['observed_p']:
        clear_line2D(fig, line, ax_p, redraw=False)

    ### Plot the observations that match the comb
    del plotted_lines['observed_p'][:]
    x = comb_params['match']
    y = np.repeat(0.6, x.size)
    line, = ax_p.plot(x, y, color='limegreen', marker=7, alpha=1, zorder=2, picker=5, ls='None')
    plotted_lines['observed_p'].append(line)

    fig.canvas.draw_idle()    

##############################################################################


def explore_results2(event):

    global P0_grid, dP0_grid, Sigma_grid, args
    
    # Grid of P0, dP0, Sigma
    P0_grid = np.arange(comb_params['P0']-50*P0_resolution,
                        comb_params['P0']+50*P0_resolution,
                        P0_resolution)
    dP0_grid = np.arange(comb_params['dP0']-50*dP0_resolution/10,
                         comb_params['dP0']+50*dP0_resolution/10,
                         dP0_resolution/10)
    Sigma_grid = np.arange(comb_params['Sigma']-50*Sigma_resolution,
                           comb_params['Sigma']+50*Sigma_resolution,
                           Sigma_resolution)    
    
    results = np.empty( [P0_grid.size,
                         dP0_grid.size,
                         Sigma_grid.size] )
    
    # Fit parameters
    P_obs        = df.query('selection==1').period.values
    e_P_obs      = df.query('selection==1').e_period.values
    A_obs        = df.query('selection==1').amp.values
    weights_obs  = A_obs/A_obs.max()
    weights_obs /= weights_obs.sum() # normalize the weights
    
    args = (comb_params['nr'], comb_params['nl'], P_obs, weights_obs, e_P_obs)
    
    jit_compute_S_on_grid = jit(compute_S_on_grid, nopython=True)
    jit_compute_S_on_grid(results)

#     results = compute_S_on_grid(results)
    
    ### Plot P0 vs dP0 <--------------------------------------------------
    Z = np.minimum.reduce(results, axis=2)
    Z = np.log(Z)

    levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('terrain')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    ax      = ax_res_P0dP0
    ax_cbar = ax_res_P0dP0_cbar
    ax.clear()
    ax_cbar.clear()
    
    X, Y = np.meshgrid(P0_grid, dP0_grid)

    #         ax.clear()
    cf = ax.contourf(X.T, Y.T, Z, levels=levels, cmap=cmap)
    #         cbar_axes['P0dP0'].clear()
    cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')
    ax_cbar.xaxis.tick_top()
    ax_cbar.xaxis.set_label_position('top') 
    ax_cbar.set_xlabel(f'log(S)')

    
    # Plot bes-fit results
    (color, ls, lw) = ('r', 'solid', 0.5)
    ax.axvline(comb_params['P0'],  color=color, ls=ls, lw=lw)
    ax.axhline(comb_params['dP0'], color=color, ls=ls, lw=lw)

    ax.set_xlabel('$P_0$')
    ax.set_ylabel('$\Delta P_0$')
    
    xlim = ax.get_xlim()
    
    ### Plot P0
    ax = ax_res_P0
    ax.clear()
    
    ax.plot(P0_grid,
            np.minimum.reduce(results, axis=(1,2)),
            ls='solid', lw=1, marker='.', markersize=1, color='k')
    
    ax.axvline(comb_params['P0'], color='r', ls='solid', lw=1)
    ax.axvspan(comb_params['P0']-comb_params['e_P0'],
               comb_params['P0']+comb_params['e_P0'],
               color='red', ls='dashed', lw=1, alpha=0.5)
    
    ax.set_xlabel('$P_0$')
    ax.set_ylabel('min $S$')
    ax.set_xlim(xlim)
    
    
    ### Plot dP0 vs Sigma <----------------------
    Z = np.minimum.reduce(results, axis=0)
    Z = np.log(Z)

    levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('terrain')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    ax      = ax_res_dP0Sigma
    ax_cbar = ax_res_dP0Sigma_cbar
    ax.clear()
    ax_cbar.clear()
    

    X, Y = np.meshgrid(dP0_grid,Sigma_grid)

    #         ax.clear()
    cf = ax.contourf(X.T, Y.T, Z, levels=levels, cmap=cmap)
    #         cbar_axes['P0dP0'].clear()
    cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')
    ax_cbar.xaxis.tick_top()
    ax_cbar.xaxis.set_label_position('top') 
    ax_cbar.set_xlabel(f'log(S)')
    
    
    # Plot bes-fit results
    (color, ls, lw) = ('r', 'solid', 0.5)
    ax.axvline(comb_params['dP0'],  color=color, ls=ls, lw=lw)
    ax.axhline(comb_params['Sigma'], color=color, ls=ls, lw=lw)

    ax.set_xlabel('$\Delta P_0$')
    ax.set_ylabel('$\Sigma$')

    xlim = ax.get_xlim()
    
    ### Plot dP0
    ax = ax_res_dP0
    ax.clear()

    ax.plot(dP0_grid,
            np.minimum.reduce(results, axis=(0,2)),
            ls='solid', lw=1, marker='.', markersize=1, color='k')

    ax.axvline(comb_params['dP0'], color='r', ls='solid', lw=1)
    ax.axvspan(comb_params['dP0']-comb_params['e_dP0'],
               comb_params['dP0']+comb_params['e_dP0'],
               color='r', ls='dashed', lw=1, alpha=0.5)

    ax.set_xlabel('$\Delta P_0$')
    ax.set_ylabel('min $S$')
    ax.set_xlim(xlim)
    
    
    ### Plot Sigma vs P0 <-----------------------------------------
    Z = np.minimum.reduce(results, axis=1)
    Z = np.log(Z)

    levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('terrain')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    ax      = ax_res_SigmaP0
    ax_cbar = ax_res_SigmaP0_cbar
    ax.clear()
    ax_cbar.clear()
    
    X, Y = np.meshgrid(Sigma_grid,P0_grid)

    
    #         ax.clear()
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    #         cbar_axes['P0dP0'].clear()
    cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')
    ax_cbar.xaxis.tick_top()
    ax_cbar.xaxis.set_label_position('top') 
    ax_cbar.set_xlabel(f'log(S)')

    # Plot bes-fit results
    (color, ls, lw) = ('r', 'solid', 0.5)
    ax.axvline(comb_params['Sigma'],  color=color, ls=ls, lw=lw)
    ax.axhline(comb_params['P0'], color=color, ls=ls, lw=lw)

    ax.set_xlabel('$\Sigma$')
    ax.set_ylabel('$P_0$')
    
    xlim = ax.get_xlim()
    
    ### Plot Sigma
    ax = ax_res_Sigma
    ax.clear()

    ax.plot(Sigma_grid,
            np.minimum.reduce(results, axis=(0,1)),
            ls='solid', lw=1, marker='.', markersize=1, color='k')

    ax.axvline(comb_params['Sigma'], color='r', ls='solid', lw=1)
    ax.axvspan(comb_params['Sigma']-comb_params['e_Sigma'],
               comb_params['Sigma']+comb_params['e_Sigma'],
               color='r', ls='dashed', lw=1, alpha=0.5)

    ax.set_xlabel('$\Sigma$')
    ax.set_ylabel('min $S$')    
    ax.set_xlim(xlim)

##############################################################################
    
def compute_S_on_grid(M):
    '''compute S for all parameter space M'''
#     global P0_grid,dP0_grid,Sigma_grid, args
    for i in range(P0_grid.size):
        for j in range(dP0_grid.size):
            for k in range(Sigma_grid.size):
                M[i,j,k] = _S(P0_grid[i],dP0_grid[j],Sigma_grid[k],*args)

##############################################################################
    
def save(event):
    
    global comb_params
    
    # Create a window
    window = tk.Tk()
    # Remove window from screen (without destroying it)
    window.withdraw()
    # Tentative output name
    if TIC:
        tentative_name = f'tic{TIC}'
    else:
        tentative_name = 'pattern'
    # Input dialog
    outputname = simpledialog.askstring(title="Save results",
                                        prompt="Enter an output name without extension.",
                                        initialvalue=tentative_name)
    # Save
    if outputname:
        # Save a PDF
        fig.savefig(f'{outputname}.pdf')
        # Save the comb pattern found as a dictionary in a pickle file
        with open(f'{outputname}.pickled', 'wb') as picklefile:
            pickle.dump(comb_params, picklefile)

##############################################################################        
        
def onselect(vmin, vmax):
    
    global df, df_full, dp, scatter1, scatter2, scatter3
    
    # If interactive is on
    if ivar['keystroke'] == 'shift+control' and vmin != vmax:
        
        if df.query('period > @vmin and period < @vmax').period.size > 0:

            # Clear plot
            for line in plotted_lines['selection_p']:
                clear_line2D(fig, line, ax_p, redraw=False)
            for line in plotted_lines['obs_dp']:
                clear_line2D(fig, line, ax_dp, redraw=False)            

            i = (df.period > vmin) & (df.period < vmax) 
            df.loc[i,'selection'] = 1 
            df.loc[~i,'selection'] = 0 
            
            ### Plot prewhitening dP vs P
            del plotted_lines['obs_dp'][:]
            xlim = ax_dp.get_xlim()
            ylim = ax_dp.get_ylim()
            x = period_for_dP_plot(df.query('selection==1').period.values, mode='middle')
            y = np.diff(df.query('selection==1').period.values)
            line, = ax_dp.plot(x, y, lw=1, color='k', ls='dashed', marker='.', zorder=2, picker=5)
            plotted_lines['obs_dp'].append(line)
            ax_dp.set_xlim(xlim)
            ax_dp.set_ylim(ylim)
        
            # Update the mod slider accordingly
            if y.min() != y.max():
                slider_mod.ax.set_xlim(y.min(),y.max())

            ### Plot available periods for the fit
            del plotted_lines['selection_p'][:]
            x = df.query('selection==1').period.values
            y = np.repeat(0.2, x.size)
            line, = ax_p.plot(x, y, color='k', marker=7, alpha=1, zorder=2, picker=5, ls='None')
            plotted_lines['selection_p'].append(line)

            ### Plot not-available periods for the fit
            x = df.query('selection==0').period.values
            y = np.repeat(0.2, x.size)
            line, = ax_p.plot(x, y, color='lightgrey', marker=7, alpha=1, zorder=2, picker=5, ls='None')
            plotted_lines['selection_p'].append(line)

            ### Update ecchelle
            scatter1.remove()
            scatter2.remove()
            scatter3.remove()
            colors = [choose_color(val) for val in df.selection.values]
            scatter1 = ax_echelle.scatter((df.period)%dp - dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
            scatter2 = ax_echelle.scatter((df.period)%dp + dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
            scatter3 = ax_echelle.scatter((df.period)%dp,      df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)

            
            slider_P0.ax.set_xlim(vmin,vmax)
            ax_echelle.set_ylim(vmin,vmax)

            fig.canvas.draw()      
        
##############################################################################

def clear_line2D(Figure, Line2D, Axes, redraw=True):
    '''Check if the given Line2D is part of the axis. If yes, remove it from the axis'''
    if np.isin(Line2D, Axes.lines):
        Axes.lines.remove(Line2D)            
        if redraw == True:
            Axes.legend(loc='best', ncol=1, framealpha=0.5, fontsize=10)
            Figure.canvas.draw()

# Conversion constants:
seconds_to_days = 1/(3600*24)

# Read prewhitened frequencies after frequency analysis for all TICs
pw_all = pd.read_csv('/lhome/stefano/Documents/work/pattern_detection/lc_stitched_11+sec_ap9+pix/freqs_comb_candicates_analysed.csv')
pw_all_groups = pw_all.groupby('tic')

# Pick a TIC
TIC = 349907707

pw = pw_all_groups.get_group(TIC).reset_index(drop=True)

# Read L-S periodogram
pg_NamePatterm = '/lhome/stefano/Documents/work/pattern_detection/lc_stitched_11+sec_ap9+pix/LS_periodogram/pg_tess{TIC}_corrected_stitched.csv'
pg = pd.read_csv(pg_NamePatterm.format(TIC=TIC))

# Add columns
if not 'period' in pg:
    pg['period'] = 1/pg.freq
if not 'e_period' in pw:
    pw['e_period'] = pw['e_frequency']*pw['period']**2 	

# Define fundamental variables
df_full = pd.DataFrame({})
df_full['period']    = pw.sort_values(by=['period']).period.values
df_full['amp']       = pw.sort_values(by=['period']).amp.values
df_full['e_period']       = pw.sort_values(by=['period']).e_period.values
df_full['selection'] = 1

df = df_full.copy()

### Shared variables of interaction

ivar = {}
ivar['keystroke']        = None
ivar['pressed_button']   = None

plotted_lines = {}

selection = {}

comb_params          = {}
comb_params['P0']    = df.query('amp == amp.max()').period.values.item() #np.median(pw_cluster.period.values)
comb_params['dP0']   = np.median(np.diff(df.period.values))
comb_params['Sigma'] = 0
comb_params['nr']    = 5
comb_params['nl']    = 5

# Generate template comb
_P, _dP = pattern_period(comb_params['P0'],
                         comb_params['dP0'],
                         comb_params['Sigma'],
                         nr=comb_params['nr'],
                         nl=comb_params['nl'])

comb_params['P']  = _P
comb_params['dP'] = _dP

comb_params['data'] = df

# Periods in the data (observations after prewhitening) that agree with the comb within dP/4
comb_params['match']     = np.array([])
comb_params['e_P0']      = 0
comb_params['e_dP0']     = 0
comb_params['e_Sigma']   = 0
comb_params['mismatch']  = -1
comb_params['residuals'] = -1
comb_params['opt']       = None


### Create figure
# fig =  plt.figure(figszie=(10,14))
fig =  plt.figure(figsize=(18,16))

### Create axis grid

big_vgrid = fig.add_gridspec(5, 1, height_ratios=[1.0, 0.5, 0.5, 0.2, 0.5], hspace=0.0)

main_hgrid = big_vgrid[0].subgridspec(1, 2, width_ratios=[3,1.2], wspace=0.2)
# main_hgrid = fig.add_gridspec(1, 2, width_ratios=[3,1.2], wspace=0.2)

sub_vgrid1 = main_hgrid[0].subgridspec(3, 1, height_ratios=[0.1,1.0,1.0], hspace=0.0)
sub_vgrid2 = main_hgrid[1].subgridspec(3, 1, height_ratios=[0.1,1.0,1.0], hspace=0.0)

ax_echelle = fig.add_subplot(sub_vgrid2[1:])
ax_p       = fig.add_subplot(sub_vgrid1[0])
ax_pg      = fig.add_subplot(sub_vgrid1[1])
ax_dp      = fig.add_subplot(sub_vgrid1[2])

# Link x-axis for the periodogram and period-spacing pattern
ax_pg.get_shared_x_axes().join(ax_dp, ax_pg)
ax_pg.get_shared_x_axes().join(ax_p, ax_pg)

ax_buttons = fig.add_subplot(big_vgrid[1])

# Results plots before the last row (density plots) 
result1_hgrid = big_vgrid[2].subgridspec(1, 3, width_ratios=[1,1,1], wspace=0.3)

result1_hgrid_1 = result1_hgrid[0].subgridspec(2, 1, height_ratios=[0.1,1], hspace=0.1)
result1_hgrid_2 = result1_hgrid[1].subgridspec(2, 1, height_ratios=[0.1,1], hspace=0.1)
result1_hgrid_3 = result1_hgrid[2].subgridspec(2, 1, height_ratios=[0.1,1], hspace=0.1)

ax_res_P0dP0    = fig.add_subplot(result1_hgrid_1[1])
ax_res_dP0Sigma = fig.add_subplot(result1_hgrid_2[1])
ax_res_SigmaP0  = fig.add_subplot(result1_hgrid_3[1])
ax_res_P0dP0_cbar    = fig.add_subplot(result1_hgrid_1[0])
ax_res_dP0Sigma_cbar = fig.add_subplot(result1_hgrid_2[0])
ax_res_SigmaP0_cbar  = fig.add_subplot(result1_hgrid_3[0])

# Results plots on the last row
result2_hgrid = big_vgrid[4].subgridspec(1, 3, width_ratios=[1,1,1], wspace=0.3)

ax_res_P0    = fig.add_subplot(result2_hgrid[0])
ax_res_dP0   = fig.add_subplot(result2_hgrid[1])
ax_res_Sigma = fig.add_subplot(result2_hgrid[2])

ax_buttons.axis('off')

### Plot available periods for the fit as triangles in the top panel
plotted_lines['selection_p'] = []
x = df.query('selection==1').period.values
y = np.repeat(0.2, x.size)
line, = ax_p.plot(x, y, color='k', marker=7, alpha=1, zorder=2, picker=5, ls='None')
l2 = line
plotted_lines['selection_p'].append(line)

# Plotted range
ax_p.set_ylim(0,1)
ax_p.axis('off')



### Plot the periodogram of the light curve
x = pg.sort_values(by=['period']).period
y = pg.sort_values(by=['period']).amp
ax_pg.plot(x, y, lw=1, color='k', zorder=3)
ax_pg.get_xaxis().set_visible(False)

# Labels
ax_pg.set_ylabel('amplitude')

# Plotted range
xlim = (df.period.min()-1/365, df.period.max()+1/365)
ax_pg.set_xlim(xlim)

### Overplot prewhitening results as vertical lines on the periodogram
for p in df.period:
    ax_pg.axvline(p, color='k', ls='dotted', lw=1)

    
    
### Plot prewhitening dP vs P
plotted_lines['obs_dp'] = []
_ = df.query('selection==1').period.values
x = period_for_dP_plot(_, mode='middle')
y = np.diff(_)
line, = ax_dp.plot(x, y, lw=1, color='k', ls='dashed', marker='.', zorder=2, picker=5)
plotted_lines['obs_dp'].append(line)

# Mark zero
ax_dp.axhline(0, ls='dotted', lw=0.5, color='gray')

# Labels
ax_dp.set_xlabel('period (days)')
ax_dp.set_ylabel('$\Delta P$ (days)')



### Plot period echelle diagram
colors = [choose_color(val) for val in df.selection.values]
dp = np.median(np.diff(df.period.values))
scatter1 = ax_echelle.scatter((df.period)%dp - dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
scatter2 = ax_echelle.scatter((df.period)%dp + dp, df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
scatter3 = ax_echelle.scatter((df.period)%dp,      df.period, s=100.*(df.amp/df.amp.max()), color=colors, zorder=3, picker=5)
plotted_lines['mod_dp'] = []
line = ax_dp.axhline(dp, color='dodgerblue', lw=1, zorder=0, ls='dotted')
plotted_lines['mod_dp'].append(line)    

# Plot period echelle diagram of the comb
scatter11 = ax_echelle.scatter((comb_params['P'])%dp - dp, comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
scatter22 = ax_echelle.scatter((comb_params['P'])%dp + dp, comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
scatter33 = ax_echelle.scatter((comb_params['P'])%dp,      comb_params['P'], s=30, color='r', zorder=4, alpha=0.3, marker='*')
scatter44 = ax_echelle.scatter((comb_params['P0'])%dp,      comb_params['P0'], s=30, color='gold', zorder=4, alpha=0.3, marker='*')

# Plotted range
ax_echelle.set_xlim(-dp,2.*dp)
ax_echelle.set_ylim(df.period.min(),df.period.max())

# Separating the different echelles in the figure
plotted_lines['echelle_vline'] = []
ax_echelle.axvline(0,  ls='dashed', color='gray', lw=2, zorder=2)
line = ax_echelle.axvline(dp, ls='dashed', color='gray', lw=2, zorder=2)
plotted_lines['echelle_vline'].append(line)    

# Labels
ax_echelle.set_ylabel('period (days)')
ax_echelle.set_xlabel(f'period mod {dp:.5f} (days)')



### Plot template comb on the periodogram
plotted_lines['comb_pg'] = []
for p in comb_params['P']:
    if p != comb_params['P0']:
        line = ax_pg.axvline(p, color='r', alpha=0.3, lw=2, zorder=0)
        plotted_lines['comb_pg'].append(line)
line = ax_pg.axvline(comb_params['P0'], color='gold', alpha=0.9, lw=2, zorder=0)
plotted_lines['comb_pg'].append(line)
# plot_updated_comb(interactive=False)

### Plot template dP vs P 
plotted_lines['comb_dp'] = []
x = period_for_dP_plot(comb_params['P'], mode='middle')
y = np.diff(comb_params['P'])   
line, = ax_dp.plot(x, y, lw=1, color='r', marker='*', ls='solid', zorder=1, alpha=0.5)
plotted_lines['comb_dp'].append(line)
if comb_params['nr'] > 1:
    ind = np.abs(comb_params['P']-comb_params['P0']).argmin()
    _ = comb_params['P'][ind:ind+2]
    x = period_for_dP_plot(_, mode='middle')
    y = np.diff(_)   
    line, = ax_dp.plot(x, y, lw=1, color='gold', marker='*', ls='None', zorder=1, alpha=.5)
    plotted_lines['comb_dp'].append(line)
    

### Plot observations that match the comb
plotted_lines['observed_p'] = []


### Sliders

# Make space to place sliders 
fig.subplots_adjust(bottom=0.2, top=0.8)

# Create axes for sliders
height   = 0.01
width    = 0.775
x0       = 0.125
y0       = 0.81
hspacing = 0.015

n_sliders = 5

coords = [ (x0, y0+i*hspacing, width, height) for i in range(n_sliders) ]
slider_axes = [ fig.add_axes(c, facecolor='lightgoldenrodyellow') for c in coords]
for ax in slider_axes:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

ax_amp   = slider_axes[0]
ax_mod   = slider_axes[1]
ax_P0    = slider_axes[2]
ax_dP0   = slider_axes[3]
ax_Sigma = slider_axes[4]

# Set range for each slider
lower_limit_amp = 0
upper_limit_amp = 1

lower_limit_P0 = df.period.min()
upper_limit_P0 = df.period.max()

lower_limit_dP0 = 100*seconds_to_days
upper_limit_dP0 = 4000*seconds_to_days

_ = df.query('selection==1').period.values
lower_limit_mod = min(lower_limit_dP0, np.diff(_).min())
upper_limit_mod = max(upper_limit_dP0, np.diff(_).max())

lower_limit_Sigma = -0.3  # -0.2 #Sigma + 0.5*Sigma
upper_limit_Sigma =  0.3  #  0.2 #Sigma - 0.5*Sigma

# Set resolution or step for each slider
amp_resolution   = 0.01
mod_resolution   = 0.0001
P0_resolution    = 0.0001
dP0_resolution   = 0.0001
Sigma_resolution = 0.001

# Set initial value for each slider
amp_initial   = 0
mod_initial   = comb_params['dP0']
P0_initial    = comb_params['P0']
dP0_initial   = comb_params['dP0']
Sigma_initial = comb_params['Sigma']

slider_amp   = Slider(ax_amp,   'ampl',         lower_limit_amp,   upper_limit_amp,   valinit=amp_initial,   valfmt='%1.2f', facecolor='k', valstep=amp_resolution)
slider_mod   = Slider(ax_mod,   'mod',          lower_limit_mod,   upper_limit_mod,   valinit=mod_initial,   valfmt='%1.6f d', facecolor='dodgerblue', valstep=mod_resolution)
slider_P0    = Slider(ax_P0,    '$P_0$',        lower_limit_P0,    upper_limit_P0,    valinit=P0_initial,    valfmt='%1.6f d', facecolor='gold',  valstep=P0_resolution)
slider_dP0   = Slider(ax_dP0,   '$\Delta P_0$', lower_limit_dP0,   upper_limit_dP0,   valinit=dP0_initial,   valfmt='%1.6f d', facecolor='red',  valstep=dP0_resolution)
slider_Sigma = Slider(ax_Sigma, '$\Sigma$',     lower_limit_Sigma, upper_limit_Sigma, valinit=Sigma_initial, valfmt='%1.6f',   facecolor='red',  valstep=Sigma_resolution)

slider_amp.on_changed(update_amplitude_tolerance)
slider_mod.on_changed(update_echelle)
slider_P0.on_changed(update_comb)
slider_dP0.on_changed(update_comb)
slider_Sigma.on_changed(update_comb)

### Build interface: Buttons, checkbox and textbox

height    = 0.02
width     = 0.1
dwidth    = 0.01
button_x0 = 0.10
# button_y0 = 0.07

xpos = button_x0+0*(width+dwidth)
# ypos = 0.07
# ypos = 0.4
ypos = ax_buttons.get_position().y0+ax_buttons.get_position().height/2


button_fit  = Button(plt.axes([xpos, ypos, width, height]), 'Fit') # <<--------------|
button_fit.on_clicked(do_fit3)

xpos = button_x0+0*(width+dwidth)
button_explore  = Button(plt.axes([xpos, ypos-height, width, height]), 'Explore result') # <<--------------|
button_explore.on_clicked(explore_results2)

xpos = button_x0+1*(width+dwidth)
button_save = Button(plt.axes([xpos, ypos, width, height]), 'Save')
button_save.on_clicked(save)


### Boxes

xpos = button_x0+2*(width+dwidth)
ax_box_mismatch = plt.axes([xpos, ypos, width, height])
text_box_mismatch = TextBox(ax_box_mismatch, "", initial='0', color='lightgoldenrodyellow')
ax_box_mismatch.set_title('Mismatch', y=-1.0)

xpos = button_x0+3*(width+dwidth)
ax_box_residuals = plt.axes([xpos, ypos, width, height])
text_box_residuals = TextBox(ax_box_residuals, "", initial='0', color='lightgoldenrodyellow')
ax_box_residuals.set_title('Residuals (d)', y=-1.0)

xpos = button_x0+4*(width+dwidth)
ax_box_P0 = fig.add_axes([xpos, ypos, width, height])
text_box_P0 = TextBox(ax_box_P0, "", initial='0')
cid_box_P0 = text_box_P0.on_submit(read_box_P0)
ax_box_P0.set_title('$P_0$ (d)', y=-1.0)

xpos = button_x0+5*(width+dwidth)
ax_box_dP0 = fig.add_axes([xpos, ypos, width, height])
text_box_dP0 = TextBox(ax_box_dP0, "", initial='0')
cid_box_dP0 = text_box_dP0.on_submit(read_box_dP0)
ax_box_dP0.set_title('$\Delta P_0$ (d)', y=-1.0)

xpos = button_x0+6*(width+dwidth)
ax_box_Sigma = fig.add_axes([xpos, ypos, width, height])
text_box_Sigma = TextBox(ax_box_Sigma, "", initial='0')
cid_box_Sigma = text_box_Sigma.on_submit(read_box_Sigma)
ax_box_Sigma.set_title('$\Sigma$', y=-1.0)

# Connect ID to the plot visualization
cid_key = fig.canvas.mpl_connect('key_press_event', read_keystroke)
cid_button = fig.canvas.mpl_connect('button_press_event', read_button)
cid_pick = fig.canvas.mpl_connect('pick_event', update_selection)

# Properties of the rectangle-span area-selector
rect_props = dict(facecolor='grey', alpha=0.20) 
# Area selector
span = mwidgets.SpanSelector(ax_pg, onselect, 'horizontal',rectprops=rect_props, useblit=False) 

ax_Sigma.set_title(f'TIC {TIC}')
# fig.suptitle(f'TIC {TIC}')

# mng = plt.get_current_fig_manager()
# mng.resize(1800,800)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# ScrollableWindow(fig)
plt.show()


# Disconnect from the plot visuzlization
for cid in [ cid_key, cid_button, cid_pick, cid_box_P0, cid_box_dP0, cid_box_Sigma ]:
    fig.canvas.mpl_disconnect(cid)
{"mode":"full","isActive":false}
