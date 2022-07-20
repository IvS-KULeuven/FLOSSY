#!/usr/bin/env python3

import time
import pickle
import numpy as np
import scipy
import pandas as pd
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from astropy import units as u
# import tkinter as tk
# from tkinter import simpledialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import matplotlib.widgets as mwidgets
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as tx
from numba import jit, njit, float32, int32

def period_for_dP_plot(periods, mode='middle'):
    """Return the array of periods with one less element to enable the plot
        with its period differences 

    Args:
        periods (_type_): _description_
        mode (str, optional): _description_. Defaults to 'middle'.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if mode == 'middle':
        return (periods[1:]+periods[:-1])/2.

    elif mode == 'right':
        return periods[1:]

    elif mode == 'left':
        return periods[:-1]

    else:
        raise ValueError(
            f'`mode` is: {mode}. It has to be one of the following values: "middle", "right", "left".')

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

    indices = np.arange(-nl, nr)

    if Sigma == 0:
        P = dP0*indices + P0
        dP = np.repeat(dP0, indices.size)
        return np.vstack((P, dP))
    else:
        P = dP0 * ((1+Sigma)**indices-1)/Sigma + P0
        dP = dP0 * (1+Sigma)**indices
        return np.vstack((P, dP))

def convert_resolution(x,dx):
    y = 1/x
    dy = dx*y**2
    return y, dy

def compute_S_on_grid(M):
    '''compute S for all parameter space M'''
#     global P0_grid,dP0_grid,Sigma_grid, args
    for i in range(P0_grid.size):
        for j in range(dP0_grid.size):
            for k in range(Sigma_grid.size):
                M[i, j, k] = _S(P0_grid[i], dP0_grid[j], Sigma_grid[k], *args)

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
        weight = np.repeat(1., P_obs.size)
    if sigma is None:
        sigma = np.repeat(0., P_obs.size)

    # Iterate over the terms of the sum
    S = 0
    for p_i, w_i, sigma_i in zip(P_obs, weight, sigma):

        i = np.abs(p_i-P_model).argmin()
        p_model = P_model[i]
        dp_model = dP_model[i]

        S += w_i*(p_i-p_model)**2/(dp_model**2+sigma_i**2)

    return S

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
    return _S(*params, nr, nl, P_obs, weight, sigma)

class UserData:
    """
    Read user data and format it if needed.
    All time units (also frequnecy) will be expressed in days.
    Amplitudes will be expressed in ppt.
    """
    
    import inputs
    
    TIC = inputs.TIC
    valid_types = (u.Unit, u.core.IrreducibleUnit, u.core.CompositeUnit, u.quantity.Quantity)
    
    # Define alias and units to be used in the rest of the code. # * Do not alter
    pw_alias_and_units = {'amplitude': ('ampl',1e-3*u.dimensionless_unscaled),
                          'frequency': ('freq',1/u.day),
                          'frequency_error': ('e_freq',1/u.day)}
    pg_alias_and_units = {'amplitude': ('ampl',1e-3*u.dimensionless_unscaled),
                          'frequency': ('freq',1/u.day)}

    def parse_pw(self):

        # Read user data
        file = self.inputs.pw_file
        col_names = self.inputs.pw_col_names
        units = self.inputs.pw_units
        freq_resolution = self.inputs.pw_frequency_resolution
        alias_units = self.pw_alias_and_units
           
        # Check that user did not change full names (keys)
        expected = set(alias_units)
        given = set(col_names)
        if given != expected:
            raise ValueError(f'Corrupted keys in `pw_col_names`. Expected keys: {expected}. Got: {given}')
        given = set(units)
        if given != expected:
            raise ValueError(f'Corrupted keys in `pw_units`. They must be same ones as in `pw_col_names`. Got: {given}')
 
        # Validate units
        for k,v in units.items():
            if not isinstance(v, self.valid_types):
                raise ValueError(f'Units of {k} in the pw are not recognized. It must be an astropy units or quantities.')
        if not isinstance(freq_resolution, self.valid_types):
            raise ValueError(f'Units of freq_resolution in the pw are not recognized. It must be an astropy units or quantities.')

        # Convert astropy units to astropy quantities
        for k,v in units.items():
            if not isinstance(v,u.quantity.Quantity):
                units[k] *= 1
        if not isinstance(freq_resolution, u.quantity.Quantity):
            freq_resolution *= 1

        # Select attibuters
        self.pw_file = file
        self.pw_freq_resolution = freq_resolution

    def parse_pg(self):
        # Read user data
        file = self.inputs.pg_file
        col_names = self.inputs.pg_col_names
        units = self.inputs.pg_units
        alias_units = self.pg_alias_and_units
                                      
        # Check that user did not change full names (keys)
        expected = set(alias_units)
        given = set(col_names)
        if given != expected:
            raise ValueError(f'Corrupted keys in `pg_col_names`. Expected keys: {expected}. Got: {given}')
        given = set(units)
        if given != expected:
            raise ValueError(f'Corrupted keys in `pg_units`. They must be same ones as in `pw_col_names`. Got: {given}')
 
        # Validate units
        for k,v in units.items():
            if not isinstance(v, self.valid_types):
                raise ValueError(f'Units of {k} in the pg are not recognized. It must be an astropy units or quantities.')

        # Convert astropy units to astropy quantities
        for k,v in units.items():
            if not isinstance(v,u.quantity.Quantity):
                units[k] *= 1

        # Select attibuters
        self.pg_file = file

    def read_pw(self):
        """Read and format prewhitening data"""

        # Fetch variables
        col_names_dic = self.inputs.pw_col_names
        units_dict = self.inputs.pw_units
        target_alias_and_units = self.pw_alias_and_units
        freq_resolution = self.pw_freq_resolution
        
        # Read pw data
        usecols = [*col_names_dic.values()]
        data = pd.read_csv(self.pw_file, usecols=usecols)

        # Give standard names to the columns
        user2name = {col_names_dic[k]:k for k in col_names_dic}
        data.rename(columns=user2name, inplace=True)

        # Convert units to target ones
        convertion_factor = {}
        for k,v in units_dict.items():
            convertion_factor[k] = v.to(target_alias_and_units[k][1]).value
            data[k] *= convertion_factor[k]
        freq_resolution = freq_resolution.to(1/u.day).value

        # Overwrite standard column names with alias ones
        name2alias = {k:target_alias_and_units[k][0] for k in col_names_dic}
        data.rename(columns=name2alias, inplace=True)

        # Add columns
        data['period'] = 1/data.freq
        data['e_period'] = data.e_freq*data.period**2
        data['selection'] = 1 # Tag for posterior interactive selection

        # Sort everything by period
        data.sort_values(by=['period'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Select attibuters
        self.pw_data = data
        self.pw_freq_resolution = freq_resolution
        self.pw_convertion_factor = convertion_factor

    def read_pg(self):
        """Read and format prewhitening data"""

        # Fetch variables
        col_names_dic = self.inputs.pg_col_names
        units_dict = self.inputs.pg_units
        target_alias_and_units = self.pg_alias_and_units
        
        # Read pg data
        usecols = [*col_names_dic.values()]
        data = pd.read_csv(self.pg_file, usecols=usecols)

        # Give standard names to the columns
        user2name = {col_names_dic[k]:k for k in col_names_dic}
        data.rename(columns=user2name, inplace=True)

        # Convert units to target ones
        convertion_factor = {}
        for k,v in units_dict.items():
            convertion_factor[k] = v.to(target_alias_and_units[k][1]).value
            data[k] *= convertion_factor[k]

        # Overwrite standard column names with alias ones
        name2alias = {k:target_alias_and_units[k][0] for k in col_names_dic}
        data.rename(columns=name2alias, inplace=True)

        # Add columns
        data['period'] = 1/data.freq

        # Sort everything by period
        data.sort_values(by=['period'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Select attibuters
        self.pg_data = data
        self.pg_convertion_factor = convertion_factor
        

    def __init__(self) -> None:
        self.parse_pw()
        self.parse_pg()
        self.read_pw()
        self.read_pg()


class IPlot:
    """Class to handle the interactive plot"""

    def __init__(self, pw, pg, freq_resolution, extra_textBoxes=False, TIC=0):
        
        class Plot:
            """Namespace plot output"""
            pass
        
        self.pw = pw.copy() # TODO: Check if copy is not necessary
        self.pg = pg
        self.freq_resolution = freq_resolution
        self.extra_textBoxes = extra_textBoxes
        self.tic = int(TIC)
        
        # Parse the raw data
        self.parse_pw()

        # Generate fig, its axes
        self.layout()
        self.format()
        
        # Make plots
        # colorOnOff = {0: 'lightgrey', 1: 'k'}
        self._colorOnOff = pd.Series(data=['lightgrey', 'k'], index=[0,1])

        # Plot containers
        self.plots = Plot() 
        self.plots.echelle_vline1 = None
        self.plots.echelle_vline2 = None
        self.plots.p_scatter_0 = None
        self.plots.p_scatter_1 = None
        self.plots.p_scatter_2 = None
        self.plots.p_scatter_3 = None
        self.plots.p_lines = None
        self.plots.data2Dline_ax_pg = None
        self.plots.PSP_pg_lines1 = None
        self.plots.PSP_pg_lines2 = None
        self.plots.PSP_pg_vline = None
        self.plots.PSP_dp_lines = None
        self.plots.PSP_dp_dot = None
        self.plots.PSP_echelle_scatter_1 = None
        self.plots.PSP_echelle_scatter_2 = None
        self.plots.PSP_echelle_scatter_3 = None
        self.plots.dp_hline = None
        self.plots.matches_scatter = None
        self.plots.cf = None
        
        # Interavtivity
        self.fitted_pw = None
        self.fitted_PSP = None
        self.enableSliders()
        self.picked_object = None
        self.mouse_down = False
        self.interaction = 'no'
        self.ls_dp = 'dashed'
        self.window_pmin = self.pw.period.min()
        self.window_pmax = self.pw.period.max()
        # self.fig = plt.gcf() if fig is None else fig
        # self.ax = self.fig.gca()
        self.connections = ()
        self.connections_textBoxes = []
        self.selectionSpan()
        
        self.plot_data(plot_pg=True)
        self.plot_module_p()
        self.plot_PSP()
        # Ensure that the plot is displayed
        p = self.pw.period.values
        xrange = p.max()-p.min()
        xrange *= 1.2
        xmid = (p.max()+p.min())/2
        self.axs.pg.set_xlim(xmid-xrange/2, xmid+xrange/2)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        """ Install the event handlers for the plot. """
        print('CONECTED')
        self.connections = [self.fig.canvas.mpl_connect('button_press_event', self.on_click),
                            self.fig.canvas.mpl_connect('pick_event', self.on_pick),
                            self.fig.canvas.mpl_connect('key_press_event', self.on_key)]

    def disconnect(self):
        """ Uninstall the event handlers for the plot. """
        for connection in self.connections+self.connections_textBoxes:
            self.fig.canvas.mpl_disconnect(connection)

    def on_click(self, event):

        if not self.interaction == 'pick':
            return

        # Periodogram
        if event.inaxes == self.axs.pg:
            nl = self.PSP.nl
            nr = self.PSP.nr
            # Add 1 line
            if event.button == 1:
                # click to the left of the linear PSP
                if event.xdata < self.PSP.P0:
                    nl += 1
                else:
                    nr += 1
            # Remove 1 line
            if event.button == 3:
                # click to the left of the linear PSP
                if event.xdata < self.PSP.P0:
                    nl -= 1
                else:
                    nr -= 1
            # Add/Remove 5 lines
            if event.button == 2:
                # click inside or outside of the left side of the linear PSP
                if event.xdata < self.PSP.P0:
                    nl += 5 if event.xdata < self.PSP.p[0] else -5
                else:
                    nr += 5 if event.xdata > self.PSP.p[-1] else -5

            # Ensure acceptable value
            self.PSP.nl = max(nl, 0)
            self.PSP.nr = max(nr, 2)
                
            # self.PSP.nr = max(self.PSP.nr, 1) # 21 OK
            # self.PSP.nl = max(self.PSP.nl, 1)

        # Plot dP vs P
        if event.inaxes == self.axs.dp:
            P0 = event.xdata - event.ydata/2
            dP0 = event.ydata
            # Set P0 and dP0
            if event.button == 1:
                self.sliders.P0.set_val(P0)
                self.PSP.P0 = P0
                self.sliders.dP0.set_val(dP0)
                self.PSP.dP0 = dP0
            # Set Sigma
            if event.button == 3:
                if self.PSP.P0 != P0:
                    Sigma = (self.PSP.dP0-dP0) / (self.PSP.P0-P0)  # slope formula
                    self.sliders.Sigma.set_val(Sigma)
                    self.PSP.Sigma = Sigma

        # Correlation plot dP0 and Sigma
        if event.inaxes == self.axs.dP0Sigma:
            # Set dP0 and Sigma
            if event.button == 1:
                self.sliders.dP0.set_val(event.xdata)
                self.PSP.dP0 = event.xdata
                self.sliders.Sigma.set_val(event.ydata)
                self.PSP.Sigma = event.ydata

        # Correlation plot P0 and dP0
        if event.inaxes == self.axs.P0dP0:
            # Set P0 and dP0
            if event.button == 1:
                self.sliders.P0.set_val(event.xdata)
                self.PSP.P0 = event.xdata
                self.sliders.dP0.set_val(event.ydata)
                self.PSP.dP0 = event.ydata

        # Correlation plot Sigma and P0
        if event.inaxes == self.axs.SigmaP0:
            # Set Sigma and P0
            if event.button == 1:
                self.sliders.Sigma.set_val(event.xdata)
                self.PSP.Sigma = event.xdata
                self.sliders.P0.set_val(event.ydata)
                self.PSP.P0 = event.ydata

        # Plot P0
        if event.inaxes == self.axs.P0:
            # Set P0
            if event.button == 1:
                self.sliders.P0.set_val(event.xdata)
                self.PSP.P0 = event.xdata

        # Plot dP0
        if event.inaxes == self.axs.dP0:
            # Set dP0
            if event.button == 1:
                self.sliders.dP0.set_val(event.xdata)
                self.PSP.dP0 = event.xdata

        # Plot Sigma
        if event.inaxes == self.axs.Sigma:
            # Set Sigma
            if event.button == 1:
                self.sliders.Sigma.set_val(event.xdata)
                self.PSP.Sigma = event.xdata

        # Generate and plot linear PSP
        self.PSP.generatePSP()
        self.plot_PSP()


    def read_box_P0(self,text):
        P0 = np.float(text)
        self.sliders.P0.set_val(P0)
        # self.PSP.P0 = P0
        # plot_comb(interactive=True)

    def read_box_dP0(self,text):
        dP0 = np.float(text)
        self.sliders.dP0.set_val(dP0)
        # self.PSP.dP0 = dP0
        # plot_comb(interactive=True)

    def read_box_Sigma(self,text):
        Sigma = np.float(text)
        self.sliders.Sigma.set_val(Sigma)
        # self.PSP.Sigma = Sigma
        # plot_comb(interactive=True)

    def on_pick(self, event):
        
        in_ax_p = event.mouseevent.inaxes == self.axs.p
        in_ax_echelle = event.mouseevent.inaxes == self.axs.echelle
        pick_mode = self.interaction == 'pick'

        print('picker 0')

        if not pick_mode:
            return
        if not (in_ax_p or in_ax_echelle):
            return
        
        pw = self.pw
        
        print('picker 1')
        
        # Swap picked period
        if in_ax_p:
            i = (pw.period-event.mouseevent.xdata).abs().argmin()
        elif in_ax_echelle:
            i = (pw.period-event.mouseevent.ydata).abs().argmin()
        swap = {0:1, 1:0}
        pw.loc[i,'selection'] = swap[pw.loc[i,'selection']]

        print('picker 2')

        # Update data plots
        self.plot_data(echelle_keep_xlim_ylim=True)

        print('picker 3')

        # Update the module period slider accordingly
        dp = np.diff(pw.query('selection==1').period.values)
        if dp.min() != dp.max():
            ax = self.axs.sliders.module_p
            slider = self.sliders.module_p
            valinit = slider.val
            self.update_slider(ax, slider, dp.min(), dp.max(), valinit)

    def on_key(self, event):
        # Interactive modes
        if event.key=='enter':
            self.interaction = 'pick'
        elif event.key=='shift+enter':
            self.interaction = 'span'
        else:
            self.interaction = 'no'
        # Toggle line style for dp plot
        if event.key=='x':
            swap = {'dashed':'None', 'None':'dashed'}
            self.ls_dp = swap[self.ls_dp]
            self.plot_dp()
            self.fig.canvas.draw_idle()



    def save(self, event):

        if self.fitted_pw is None:
            return
        #or not self.fitted_PSP:
        fitted_pw = self.fitted_pw.copy()
        fitted_PSP = self.fitted_PSP.copy()
        
        outputname = f'tic{self.tic}'
        
        # Save
        if outputname:
            # Save a PDF
            self.fig.savefig(f'{outputname}.pdf', bbox_inches='tight')
            # Save the comb pattern found as a dictionary in a pickle file
            # with open(f'{outputname}.pickled', 'wb') as picklefile:
            #     pickle.dump(comb_params, picklefile)
            cols = ['period', 'e_period', 'freq', 'e_freq', 'match', 'imatch']
            fitted_pw[cols].to_csv(f'{outputname}_obs.csv', index=False)
            cols = ['i', 'period', 'period_spacing', 'missing']
            fitted_PSP[cols].to_csv(f'{outputname}_PSP.csv', index=False)


    def explore_results(self,event):

        global P0_grid, dP0_grid, Sigma_grid, args

        # Grid of P0, dP0, Sigma
        P0_resolution = 0.0001 # TODO: Express as a day-like resolution
        dP0_resolution = 0.0001 # TODO: Express as a day-like resolution
        Sigma_resolution = 0.001 # TODO: Express as a day-like resolution
        P0_grid = np.arange(self.PSP.P0-50*P0_resolution,
                            self.PSP.P0+50*P0_resolution,
                            P0_resolution)
        dP0_grid = np.arange(self.PSP.dP0-50*dP0_resolution/10,
                            self.PSP.dP0+50*dP0_resolution/10,
                            dP0_resolution/10)
        Sigma_grid = np.arange(self.PSP.Sigma-50*Sigma_resolution,
                            self.PSP.Sigma+50*Sigma_resolution,
                            Sigma_resolution)

        results = np.empty([P0_grid.size,
                            dP0_grid.size,
                            Sigma_grid.size])

        # Fit parameters
        P_obs = self.pw.query('selection==1').period.values
        e_P_obs = self.pw.query('selection==1').e_period.values
        A_obs = self.pw.query('selection==1').ampl.values
        w_obs = A_obs/A_obs.max()
        w_obs /= w_obs.sum()  # normalize the weights

        args = (self.PSP.nr, self.PSP.nl, P_obs, w_obs, e_P_obs)

        jit_compute_S_on_grid = jit(compute_S_on_grid, nopython=True)
        jit_compute_S_on_grid(results)

        # Plot P0 vs dP0
        Z = np.minimum.reduce(results, axis=2)
        Z = np.log(Z)

        levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('terrain')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ax = self.axs.P0dP0
        ax_cbar = self.axs.P0dP0_cbar
        ax.clear()
        ax_cbar.clear()

        X, Y = np.meshgrid(P0_grid, dP0_grid)

        cf = ax.contourf(X.T, Y.T, Z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')
        self.plots.cf = cf

        # Plot bes-fit results
        (color, ls, lw) = ('r', 'solid', 0.5)
        ax.axvline(self.PSP.P0,  color=color, ls=ls, lw=lw)
        ax.axhline(self.PSP.dP0, color=color, ls=ls, lw=lw)

        xlim = ax.get_xlim()

        # Plot P0
        ax = self.axs.P0
        ax.clear()

        ax.plot(P0_grid,
                np.minimum.reduce(results, axis=(1, 2)),
                ls='solid', lw=1, marker='.', markersize=1, color='k')

        ax.axvline(self.PSP.P0, color='r', ls='solid', lw=1)

        ax.set_xlim(xlim)

        # Plot dP0 vs Sigma
        Z = np.minimum.reduce(results, axis=0)
        Z = np.log(Z)

        levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('terrain')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ax = self.axs.dP0Sigma
        ax_cbar = self.axs.dP0Sigma_cbar
        ax.clear()
        ax_cbar.clear()

        X, Y = np.meshgrid(dP0_grid, Sigma_grid)

        cf = ax.contourf(X.T, Y.T, Z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')

        # Plot bes-fit results
        (color, ls, lw) = ('r', 'solid', 0.5)
        ax.axvline(self.PSP.dP0,  color=color, ls=ls, lw=lw)
        ax.axhline(self.PSP.Sigma, color=color, ls=ls, lw=lw)

        xlim = ax.get_xlim()

        # Plot dP0
        ax = self.axs.dP0
        ax.clear()

        ax.plot(dP0_grid,
                np.minimum.reduce(results, axis=(0, 2)),
                ls='solid', lw=1, marker='.', markersize=1, color='k')

        ax.axvline(self.PSP.dP0, color='r', ls='solid', lw=1)
        # ax.axvspan(self.PSP.dP0-self.PSP.e_dP0,
        #         self.PSP.dP0+self.PSP.e_dP0,
        #         color='r', ls='dashed', lw=1, alpha=0.5)

        ax.set_xlim(xlim)

        # Plot Sigma vs P0 
        Z = np.minimum.reduce(results, axis=1)
        Z = np.log(Z)

        levels = MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('terrain')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ax = self.axs.SigmaP0
        ax_cbar = self.axs.SigmaP0_cbar
        ax.clear()
        ax_cbar.clear()

        X, Y = np.meshgrid(Sigma_grid, P0_grid)

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax, cax=ax_cbar, orientation='horizontal')

        # Plot bes-fit results
        (color, ls, lw) = ('r', 'solid', 0.5)
        ax.axvline(self.PSP.Sigma,  color=color, ls=ls, lw=lw)
        ax.axhline(self.PSP.P0, color=color, ls=ls, lw=lw)

        xlim = ax.get_xlim()

        # Plot Sigma
        ax = self.axs.Sigma
        ax.clear()

        ax.plot(Sigma_grid,
                np.minimum.reduce(results, axis=(0, 1)),
                ls='solid', lw=1, marker='.', markersize=1, color='k')

        ax.axvline(self.PSP.Sigma, color='r', ls='solid', lw=1)

        ax.set_xlim(xlim)
        self.format(noSliders=True)
        self.fig.canvas.draw_idle()

    def fitPSP(self,event):
        
        # Clear result previous fit
        self.fitted_pw = None
        self.fitted_PSP = None
        
        # Data
        fitted_pw = self.pw.query('selection==1').copy()
        fitted_pw.reset_index(inplace=True, drop=True)
        fitted_pw['match'] = 0
        p = fitted_pw.period.values
        e_p = fitted_pw.e_period.values
        ampl = fitted_pw.ampl.values
        freq_resolution = self.freq_resolution
        
        # weights
        w = ampl/ampl.max()
        w /= w.sum()  # normalize

        # Minimization parameters
        x0 = [self.PSP.P0, self.PSP.dP0, self.PSP.Sigma]
        
        # Bounds for the fit
        # P0_bounds = (None, None) 
        # dP0_bounds = (None, None)
        # Sigma_bounds = (None, None)
        
        # Bounds P0
        delta = self.PSP.dP0/2
        vmin = self.PSP.P0-delta
        vmin = max(vmin, self.range_P0[0])
        vmax = self.PSP.P0+delta
        vmax = min(vmax, self.range_P0[1])
        P0_bounds = (vmin, vmax)
        # Bounds dP0
        vmin = self.PSP.dP0/2
        vmin = max(vmin, self.range_dP0[0])
        vmax = self.PSP.dP0*2
        vmax = min(vmax, self.range_dP0[1])
        dP0_bounds = (vmin, vmax)
        # Bounds Sigma
        delta = 0.15
        vmin = self.PSP.Sigma-delta
        vmin = max(vmin, self.range_Sigma[0])
        vmax = self.PSP.Sigma+delta
        vmax = min(vmax, self.range_Sigma[1])
        Sigma_bounds = (vmin, vmax)
        # Bounds for the fit
        bounds = [P0_bounds, dP0_bounds, Sigma_bounds]
        # Arguments for the fit
        args = (self.PSP.nr, self.PSP.nl, p, w, e_p)

        results = minimize(S, x0, args=args, bounds=bounds)

        self.PSP.P0 = results.x[0]
        self.PSP.dP0 = results.x[1]
        self.PSP.Sigma = results.x[2]
        self.PSP.generatePSP() # ? Does trigger plot?

        # Update slider P0 (plot automatically updated)
        ax = self.axs.sliders.P0
        slider = self.sliders.P0
        valinit = self.PSP.P0
        self.update_slider(ax, slider, slider.valmin, slider.valmax, valinit)
        # Update slider dP0 (plot automatically updated)
        ax = self.axs.sliders.dP0
        slider = self.sliders.dP0
        valinit = self.PSP.dP0
        self.update_slider(ax, slider, slider.valmin, slider.valmax, valinit)
        # Update slider Sigma (plot automatically updated)
        ax = self.axs.sliders.Sigma
        slider = self.sliders.Sigma
        valinit = self.PSP.Sigma
        self.update_slider(ax, slider, slider.valmin, slider.valmax, valinit)

        # Find data that matches PSP within the data frequency resolution
        def match(p_data, periods_PSP):
            dp = np.abs(p_data-periods_PSP).min()
            _, df = convert_resolution(p_data,dp)
            return 1 if df < self.freq_resolution else 0
        fitted_pw['match'] = fitted_pw.period.apply(match, args=(self.PSP.p,))
        
        # Find the index in the PSP of fitted_pw
        def imatch(p_data, periods_PSP):
            i = np.abs(p_data-periods_PSP).argmin()
            dp = np.abs(p_data-periods_PSP).min()
            _, df = convert_resolution(p_data,dp)
            return i if df < self.freq_resolution else -1
        fitted_pw['imatch'] = fitted_pw.period.apply(imatch, args=(self.PSP.p,))

        def missing(index, indexes):
            return 0 if index in indexes else 1
            
        fitted_PSP = pd.DataFrame({'period': self.PSP.p,
                                   'period_spacing':self.PSP.dp,
                                   'i':np.arange(self.PSP.p.size)})
        fitted_PSP['missing'] = fitted_PSP.i.apply(missing, args=(fitted_pw.imatch.values,))

        # Needed for saving output to text file
        self.fitted_pw = fitted_pw
        self.fitted_PSP = fitted_PSP
                
        # Plot the matches
        if self.plots.matches_scatter:
            self.plots.matches_scatter.remove()
        matches = fitted_pw.query('match==1').period
        if len(matches)>0:
            ax = self.axs.p
            x = matches
            y = np.repeat(0.6, len(x))
            self.plots.matches_scatter = ax.scatter(x, y, c='limegreen', marker=7, alpha=1, zorder=2, picker=5)
        # Reinitialize container
        else:
            self.plots.matches_scatter = None
            
        # Residuals
        r = []
        # Iterate over data, not PSP
        for _p in p:
            r.append(np.abs(_p-self.PSP.p).min())
        residuals = np.sum(r)

        self.textBoxes.residuals.set_val(f'{residuals:.6f}')
        self.residulas = residuals

    # Set of functions to add and remove data from axes
    def plot_data(self, plot_pg=False, echelle_keep_xlim=False, echelle_keep_ylim=False, echelle_keep_xlim_ylim=False):
        if plot_pg:
            self.plot_pg()
        self.plot_p()
        self.add_p2pg()
        self.plot_dp()
        self.plot_echelle(keep_xlim=echelle_keep_xlim, keep_ylim=echelle_keep_ylim, keep_xlim_ylim=echelle_keep_xlim_ylim)

    # Set of functions to add and remove PSP from axes
    def plot_PSP(self):
        self.PSP.add2pg()
        self.PSP.add2dp()
        self.PSP.add2echelle()

    def update_axes_echelle(self, **kwargs):
        self.plot_echelle(**kwargs)
        self.PSP.add2echelle()
        self.plot_module_p()
        self.axs.echelle.set_xlabel(f'period mod {self.module_dp:.5f} (days)') # * Repeated line in format()

    def update_slider(self, ax, slider, vmin, vmax, valinit):
        slider.valmin = vmin
        slider.valmax = vmax
        ax.set_xlim(slider.valmin,slider.valmax)
        slider.set_val(valinit)
        slider.valtext.set_text(slider.valfmt%valinit) # This triggers the slider's callback

    def onselect2(self, vmin, vmax):
        print('Span selection: ', vmin, vmax)
        if self.interaction == 'span' and vmin != vmax:
            print('print 1')
            query = 'period>@vmin and period<@vmax'
            pw = self.pw.query(query)
            if pw.period.size > 0:
                self.window_pmin = vmin
                self.window_pmax = vmax
                i = (self.pw.period > vmin) & (self.pw.period < vmax)
                self.pw.selection = 0
                self.pw.loc[i, 'selection'] = 1
                # Clear plots
                print('print 2')
                xlim = self.axs.pg.get_xlim()
                self.plot_data(echelle_keep_xlim_ylim=True)
                self.axs.pg.set_xlim(xlim)
                print('print 4')
                # Update slider P0
                ax = self.axs.sliders.P0
                slider = self.sliders.P0
                valinit = slider.val
                self.update_slider(ax, slider, vmin, vmax, valinit)
                # Update slider module period
                print('print 5')
                p = self.pw.query('selection==1').period.values
                dp = np.diff(p)
                ax = self.axs.sliders.module_p
                slider = self.sliders.module_p
                vmin = dp.min()/4
                vmax = dp.max()*2
                valinit = (vmin+vmax)/2
                self.update_slider(ax, slider, vmin, vmax, valinit)
                # Update slider dp
                print('print 5')
                p = self.pw.query('selection==1').period.values
                dp = np.diff(p)
                ax = self.axs.sliders.dP0
                slider = self.sliders.dP0
                vmin = dp.min()/4
                vmax = dp.max()*2
                valinit = (vmin+vmax)/2
                self.update_slider(ax, slider, vmin, vmax, valinit)
    
    def selectionSpan(self):        
        # Properties of the rectangle-span area-selector
        rect_props = dict(facecolor='grey', alpha=0.20)
        # Area selector
        self.span = mwidgets.SpanSelector(self.axs.pg, self.onselect2, 'horizontal', rectprops=rect_props, useblit=False)

    def ampl_threshold(self,val):
        # Find periods above threshold and within the window
        pw = self.pw
        pmin = self.window_pmin
        pmax = self.window_pmax
        query = 'period>@pmin and period<@pmax'
        ampl_max = pw.query(query).ampl.max()
        i = (pw.ampl/ampl_max >= val) & (pw.period >= pmin) & (pw.period <= pmax)
        # Leave at least two periods
        if pw[i].period.size >= 2:
            # Apply threshold
            pw.selection = 0
            pw.loc[i, 'selection'] = 1
            # Update p in plot
            xlim = self.axs.pg.get_xlim()
            self.plot_data(echelle_keep_xlim_ylim=True)
            self.axs.pg.set_xlim(xlim)
        
    def sliderAction_P0(self,val):
        print('Slider action P0 was riggered')
        self.PSP.P0 = val
        self.PSP.generatePSP()
        self.plot_PSP()
    def sliderAction_dP0(self,val):
        self.PSP.dP0 = val
        self.PSP.generatePSP()
        self.plot_PSP()
    def sliderAction_Sigma(self,val):
        self.PSP.Sigma = val
        self.PSP.generatePSP()
        self.plot_PSP()

    def update_module_p(self,val):
        self.module_dp = val
        self.update_axes_echelle(keep_ylim=True)

    def enableSliders(self):
        self.sliders.ampl.on_changed(self.ampl_threshold)
        self.sliders.module_p.on_changed(self.update_module_p)
        self.sliders.P0.on_changed(self.sliderAction_P0)
        self.sliders.dP0.on_changed(self.sliderAction_dP0)
        self.sliders.Sigma.on_changed(self.sliderAction_Sigma)

    def LinearPSP(self,P0,dP0,Sigma=0,nr=5,nl=5) -> None:
        
        class PSPspace:
            pass
        
        self.PSP= PSPspace()
        self.PSP.P0 = P0
        self.PSP.dP0 = dP0
        self.PSP.Sigma = Sigma
        self.PSP.nr = nr
        self.PSP.nl = nl
        self.PSP.generatePSP = self.generatePSP
        self.PSP.add2pg = self.add2pg
        self.PSP.add2dp = self.add2dp
        self.PSP.add2echelle = self.add2echelle
        self.PSP.generatePSP()
        
    def generatePSP(self):
        self.PSP.p, self.PSP.dp = pattern_period(self.PSP.P0,self.PSP.dP0,self.PSP.Sigma,self.PSP.nr,self.PSP.nl)

    def add2echelle(self):
        # Clear plot if
        if self.plots.PSP_echelle_scatter_1:
            self.plots.PSP_echelle_scatter_1.remove()
        if self.plots.PSP_echelle_scatter_2:
            self.plots.PSP_echelle_scatter_2.remove()
        if self.plots.PSP_echelle_scatter_3:
            self.plots.PSP_echelle_scatter_3.remove()
        ax = self.axs.echelle
        p = self.PSP.p
        module_dp = self.module_dp
        color = 'r'
        size = 30
        self.plots.PSP_echelle_scatter_1 = ax.scatter(p%module_dp-module_dp, p, s=size, color=color, zorder=3, picker=5)
        self.plots.PSP_echelle_scatter_2 = ax.scatter(p%module_dp+module_dp, p, s=size, color=color, zorder=3, picker=5)
        self.plots.PSP_echelle_scatter_3 = ax.scatter(p%module_dp, p, s=size, color=color, zorder=3, picker=5)

    def add2pg(self):
        # Clear plot if
        if self.plots.PSP_pg_vline:
            self.plots.PSP_pg_vline.remove()
        # Clear plot if
        if self.plots.PSP_pg_lines1:            
            for line in self.plots.PSP_pg_lines1:
                line.remove()
        # Clear plot if
        if self.plots.PSP_pg_lines2:            
            for line in self.plots.PSP_pg_lines2:
                line.remove()
                
        ax = self.axs.pg
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        p = self.PSP.p
        P0 = self.PSP.P0
        self.plots.PSP_pg_lines1 = ax.plot(np.repeat(p, 3), np.tile([0, 1, np.nan], len(p)), color='r', alpha=0.3, lw=2, zorder=0, transform=trans)
        # Overplot P0 with a different color
        self.plots.PSP_pg_vline = ax.axvline(P0, color='gold', alpha=0.9, lw=2, zorder=0)

        # Unresolved frequencies
        freq_resolution = self.freq_resolution
        dp = self.PSP.dp
        freq, dfreq = convert_resolution(p,dp)
        unresolved_p = p[dfreq <= freq_resolution]
        if len(unresolved_p) > 0:
            self.plots.PSP_pg_lines2 = ax.plot(np.repeat(unresolved_p, 3), np.tile([0, 1, np.nan], len(unresolved_p)), color='darkviolet', alpha=1, lw=2, zorder=0, ls='-', transform=trans)
        else:
            # Reinitialize container
            self.plots.PSP_pg_lines2 = None
            
    def add2dp(self):
        # Clear plot if
        if self.plots.PSP_dp_lines:            
            for line in self.plots.PSP_dp_lines:
                line.remove()
        if self.PSP.nr >= 2: # Ensure that there is at least two periods to the right
            if self.plots.PSP_dp_dot:            
                for line in self.plots.PSP_dp_dot:
                    line.remove()

        ax = self.axs.dp
        p = self.PSP.p
        x = period_for_dP_plot(p, mode='middle')
        y = np.diff(p)
        self.plots.PSP_dp_lines = ax.plot(x, y, lw=1, color='r', marker='*', ls='solid', zorder=1, alpha=0.5)
        # self.outer.plots.PSP_echelle_scatter_3 = ax.scatter(p%module_dp, p, s=size, color=color, zorder=3, picker=5)
        # Overplot dp associated with P0 with a different color
        if self.PSP.nr >= 1:
            i = np.abs(self.PSP.p-self.PSP.P0).argmin()
            period_pair = self.PSP.p[i:i+2]
            x = period_for_dP_plot(period_pair, mode='middle')
            y = np.diff(period_pair)
            self.plots.PSP_dp_dot = ax.plot(x, y, lw=1, color='gold', marker='*', ls='None', zorder=1, alpha=0.5)

    def parse_pw(self):
        # Estimate a module dp
        self.module_dp = np.median(np.diff(self.pw.period.values))
        # Estimate a linear PSP of 10 periods around the dominant period 
        self.dominant_p = self.pw.query('ampl == ampl.max()').period.values.item()
        # self.PSP = self.LinearPSP(self,self.dominant_p,self.module_dp)
        # self.PSP = self.LinearPSP(self.dominant_p,self.module_dp)
        self.LinearPSP(self.dominant_p,self.module_dp)
    
    def format(self, noSliders=False):
        """Format the layout by adding label and tweaks to the axes"""
        def fig_and_axs():
            
            # Link axis
            def xlim_to_ylim_echelle(event_ax, _self=self):
                _self.axs.echelle.set_ylim(event_ax.get_xlim())
            self.axs.dp.sharex(self.axs.pg)
            self.axs.p.sharex(self.axs.pg)
            self.axs.pg.callbacks.connect('xlim_changed', xlim_to_ylim_echelle)
            self.axs.dp.callbacks.connect('xlim_changed', xlim_to_ylim_echelle)
            
            # Labels
            self.axs.pg.set_ylabel('amplitude (ppt)')
            self.axs.dp.set_xlabel('period (days)')
            self.axs.dp.set_ylabel('$\Delta P$ (days)')
            self.axs.echelle.set_ylabel('period (days)')
            self.axs.echelle.set_xlabel(f'period mod {self.module_dp:.5f} (days)') # * Repeated line in update_echelle()
            self.axs.P0dP0.set_xlabel('$P_0$')
            self.axs.P0dP0.set_ylabel('$\Delta P_0$')
            self.axs.P0.set_xlabel('$P_0$')
            self.axs.P0.set_ylabel('min $S$')
            self.axs.dP0Sigma.set_xlabel('$\Delta P_0$')
            self.axs.dP0Sigma.set_ylabel('$\Sigma$')
            self.axs.dP0.set_xlabel('$\Delta P_0$')
            self.axs.dP0.set_ylabel('min $S$')       
            self.axs.SigmaP0.set_xlabel('$\Sigma$')
            self.axs.SigmaP0.set_ylabel('$P_0$')
            self.axs.Sigma.set_xlabel('$\Sigma$')
            self.axs.Sigma.set_ylabel('min $S$')

            # Prune y axis pg and dp
            locator=MaxNLocator(prune=None, nbins=4)
            self.axs.pg.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=3)
            self.axs.dp.yaxis.set_major_locator(locator)
            # Prune x axis pg and dp
            locator=MaxNLocator(prune='both', nbins=6)
            self.axs.dp.xaxis.set_major_locator(locator)
            # Prune echelle
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.echelle.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.echelle.yaxis.set_major_locator(locator)

            # # Prune x axis Landspace
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.P0dP0.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.dP0Sigma.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.SigmaP0.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.P0.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.dP0.xaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=5)
            self.axs.Sigma.xaxis.set_major_locator(locator)
            # # Prune y axis Landspace
            locator=MaxNLocator(prune='both', nbins=4)
            self.axs.P0dP0.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=4)
            self.axs.dP0Sigma.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='both', nbins=4)
            self.axs.SigmaP0.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='upper', nbins=4)
            self.axs.P0.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='upper', nbins=4)
            self.axs.dP0.yaxis.set_major_locator(locator)
            locator=MaxNLocator(prune='upper', nbins=4)
            self.axs.Sigma.yaxis.set_major_locator(locator)

            # Visibility
            self.axs.allButtons.axis('off')
            self.axs.p.axis('off')
            self.axs.pg.get_xaxis().set_visible(False)

            # Color bars
            locator=MaxNLocator(prune='both', nbins=5)
            # P0dP0_cbar
            self.axs.P0dP0_cbar.xaxis.tick_top()
            self.axs.P0dP0_cbar.xaxis.set_label_position('top')
            self.axs.P0dP0_cbar.set_xlabel(f'log(S)')
            self.axs.P0dP0_cbar.xaxis.set_major_locator(locator)
            # dP0Sigma_cbar
            self.axs.dP0Sigma_cbar.xaxis.tick_top()
            self.axs.dP0Sigma_cbar.xaxis.set_label_position('top')
            self.axs.dP0Sigma_cbar.set_xlabel(f'log(S)')
            self.axs.dP0Sigma_cbar.xaxis.set_major_locator(locator)
            # SigmaP0_cbar
            self.axs.SigmaP0_cbar.xaxis.tick_top()
            self.axs.SigmaP0_cbar.xaxis.set_label_position('top')
            self.axs.SigmaP0_cbar.set_xlabel(f'log(S)')
            self.axs.SigmaP0_cbar.xaxis.set_major_locator(locator)
 
            # Title
            self.axs.p.set_title(f'TIC {self.tic}', y=0.6)           
            
            # Ranges
            self.axs.p.set_ylim(0, 1)
            
        def sliders():
            # Slider axes
            for slider in vars(self.axs.sliders):
                ax = getattr(self.axs.sliders, slider)
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
            # Sliders
            for slider in vars(self.sliders):
                slider = getattr(self.sliders, slider)
                l1,l2 = slider.ax.get_lines()
                l1.remove() # Remove vertical line
            
            # Apply values            
            def apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep):
                slider.valmin = vmin
                slider.valmax = vmax
                ax.set_xlim(slider.valmin,slider.valmax)
                slider.label.set_text(label)
                slider.set_val(valinit)
                # valinit = valfmt%self.PSP.P0
                # slider.val(valinit)
                slider.valfmt = valfmt
                slider.valtext.set_text(valfmt%valinit)
                # ax.set_facecolor(facecolor)
                slider.poly.set_fc(facecolor)
                slider.valstep = valstep

            # Amplitude
            ax = self.axs.sliders.ampl
            slider = self.sliders.ampl
            label='amplitude (%)' # OK
            vmin = 0.0 # OK
            vmax = 1.0 # OK
            valfmt = '%1.2f' # OK
            valinit = 0 # OK
            # valinit = valfmt%0 # OK
            facecolor = 'k' # OK
            valstep = 0.01 # From 1% to 100% # OK      
            apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep)
            # Module P
            ax = self.axs.sliders.module_p
            slider = self.sliders.module_p
            label='modulo'
            p = self.pw.period.values
            vmin = (100*u.s).to(u.day).value # not expected under 100 seconds
            vmax = (4000*u.s).to(u.day).value # not expected over 4000 second
            valfmt = '%1.6f d' # TODO: Give physical meaning to this
            valinit = self.PSP.dP0
            # valinit = valfmt%self.PSP.dP0
            facecolor = 'dodgerblue'
            valstep = 0.0001 # TODO: Express as a day-like resolution
            apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep)
            # P0
            ax = self.axs.sliders.P0
            slider = self.sliders.P0
            label = '$P_0$'
            vmin = self.pw.period.min()
            vmax = self.pw.period.max()
            self.range_P0 = (vmin,vmax)
            valfmt = '%1.6f d' # TODO: Give physical meaning to this
            valinit = self.PSP.P0
            # valinit = valfmt%self.PSP.P0
            facecolor = 'gold'
            valstep = 0.0001 # TODO: Express as a day-like resolution
            apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep)
            # dP0
            ax = self.axs.sliders.dP0
            slider = self.sliders.dP0
            label = '$\Delta P_0$'
            vmin = (100*u.s).to(u.day).value # not expected under 100 seconds
            vmax = (4000*u.s).to(u.day).value # not expected over 4000 second
            self.range_dP0 = (vmin,vmax)
            valfmt = '%1.6f d' # TODO: Give physical meaning to this
            valinit = self.PSP.dP0
            # valinit = valfmt%self.PSP.dP0
            facecolor = 'r'
            valstep = 0.0001 # TODO: Express as a day-like resolution
            apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep)
            # Sigma
            ax = self.axs.sliders.Sigma
            slider = self.sliders.Sigma
            label = '$\Sigma$'
            vmin = -0.35 # TODO: Give physical meaning to this
            vmax = 0.35 # TODO: Give physical meaning to this
            self.range_Sigma = (vmin,vmax)
            valfmt = '%1.6f' # TODO: Give physical meaning to this
            valinit = 0
            # valinit = valfmt%0
            facecolor = 'r'
            valstep = 0.001 # TODO: Express as a day-like resolution
            apply_values(ax, slider, label, vmin, vmax, valinit, valfmt, facecolor, valstep)
            
        fig_and_axs()
        if not noSliders:
            sliders()
        
    def layout(self):
        """Initizlize figure and axes as attributes"""
        
        def fig_and_axs():
            class Axs:
                """Namespace for axes"""
                pass
            
            fig = plt.figure(figsize=(18, 16))
            axs = Axs()

            # Create axis grid
            main5Rows = fig.add_gridspec(5, 1, height_ratios=[1.0, 0.5, 0.5, 0.2, 0.5], hspace=0.0)
            main5Rows.update(left=0.05,right=0.98,top=0.9,bottom=0.03)

            # Row 0: Period indicator on top of pg (P), pg, dP, echelle 
            mainRow0_main2Cols = main5Rows[0].subgridspec(1, 2, width_ratios=[3, 1.2], wspace=0.2)
            mainRow0_mainCol0_main3Rows = mainRow0_main2Cols[0].subgridspec(3, 1, height_ratios=[0.1, 1.0, 1.0], hspace=0.0)
            mainRow0_mainCol1_main3Rows = mainRow0_main2Cols[1].subgridspec(3, 1, height_ratios=[0.1, 1.0, 1.0], hspace=0.0)
            axs.p = fig.add_subplot(mainRow0_mainCol0_main3Rows[0])
            axs.pg = fig.add_subplot(mainRow0_mainCol0_main3Rows[1])
            axs.dp = fig.add_subplot(mainRow0_mainCol0_main3Rows[2])
            axs.echelle = fig.add_subplot(mainRow0_mainCol1_main3Rows[1:])
            
            # Row 1: Buttons
            axs.allButtons = fig.add_subplot(main5Rows[1])

            # Row 2: Landscape with color bat at the top
            mainRow2_main3Cols = main5Rows[2].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
            # Accomodate space for landscape and color bar at the top
            mainRow2_mainCol0_main2Rows = mainRow2_main3Cols[0].subgridspec(2, 1, height_ratios=[0.1, 1], hspace=0.1)
            mainRow2_mainCol1_main2Rows = mainRow2_main3Cols[1].subgridspec(2, 1, height_ratios=[0.1, 1], hspace=0.1)
            mainRow2_mainCol2_main2Rows = mainRow2_main3Cols[2].subgridspec(2, 1, height_ratios=[0.1, 1], hspace=0.1)
            # Create axes for landscape and color bar
            axs.P0dP0 = fig.add_subplot(mainRow2_mainCol0_main2Rows[1])
            axs.dP0Sigma = fig.add_subplot(mainRow2_mainCol1_main2Rows[1])
            axs.SigmaP0 = fig.add_subplot(mainRow2_mainCol2_main2Rows[1])
            axs.P0dP0_cbar = fig.add_subplot(mainRow2_mainCol0_main2Rows[0])
            axs.dP0Sigma_cbar = fig.add_subplot(mainRow2_mainCol1_main2Rows[0])
            axs.SigmaP0_cbar = fig.add_subplot(mainRow2_mainCol2_main2Rows[0])
            
            # Row 3: Blank space used as spacer
            
            # Row 4: PDF
            mainRow4_main3Cols = main5Rows[4].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
            axs.P0 = fig.add_subplot(mainRow4_main3Cols[0])
            axs.dP0 = fig.add_subplot(mainRow4_main3Cols[1])
            axs.Sigma = fig.add_subplot(mainRow4_main3Cols[2])
            
            # Create attributes
            self.fig = fig
            self.axs = axs
            
        def sliders():
            class Sliders:
                """Namespace for sliders"""
                pass

            # Make space to place sliders
            self.fig.subplots_adjust(bottom=0.2, top=0.8)
            # sliders'dimensions
            height = 0.01
            width = 0.775
            x0 = 0.125
            y0 = 0.925 #0.81
            hspacing = 0.015
            # Number of sliders
            nSliders = 5
            # Slider positions
            coords = [(x0, y0+i*hspacing, width, height) for i in range(nSliders)]
            # Slider axes
            self.axs.sliders = Sliders()
            self.axs.sliders.ampl = self.fig.add_axes(coords[0])
            self.axs.sliders.module_p = self.fig.add_axes(coords[1])
            self.axs.sliders.P0 = self.fig.add_axes(coords[2])
            self.axs.sliders.dP0 = self.fig.add_axes(coords[3])
            self.axs.sliders.Sigma = self.fig.add_axes(coords[4])
            
            # Create sliders
            self.sliders = Sliders()
            # Amplitude
            ax = self.axs.sliders.ampl
            self.sliders.ampl = Slider(ax, '', 0, 1)
            # Module P
            ax = self.axs.sliders.module_p
            self.sliders.module_p = Slider(ax, '', 0, 1)
            # P0
            ax = self.axs.sliders.P0
            self.sliders.P0 = Slider(ax, '', 0, 1)
            # dP0
            ax = self.axs.sliders.dP0
            self.sliders.dP0 = Slider(ax, '', 0, 1)
            # Sigma
            ax = self.axs.sliders.Sigma
            self.sliders.Sigma = Slider(ax, '', 0, 1)

        def buttons():
            class Buttons:
                """Namespace for buttons"""
                pass
            class Boxes:
                """Namespace for boxes"""
                pass
            # Build interface: Buttons, checkbox and textbox

            height = 0.02
            width = 0.1
            dwidth = 0.01
            button_x0 = 0.10

            ypos = self.axs.allButtons.get_position().y0 + self.axs.allButtons.get_position().height/2
            xpos = button_x0+0*(width+dwidth)
            
            # Crate button axes and boxes axes
            self.axs.buttons = Buttons()
            self.axs.textBoxes = Boxes()
            
            self.axs.buttons.fit = Button(plt.axes([xpos, ypos, width, height]), 'Fit')
            self.axs.buttons.fit.on_clicked(self.fitPSP)

            xpos = button_x0+0*(width+dwidth)
            self.axs.buttons.explore = Button(plt.axes([xpos, ypos-height, width, height]), 'Explore')  # <<--------------|
            self.axs.buttons.explore.on_clicked(self.explore_results)

            xpos = button_x0+1*(width+dwidth)
            self.axs.buttons.save = Button(plt.axes([xpos, ypos, width, height]), 'Save')
            self.axs.buttons.save.on_clicked(self.save)

            # Create buttons and boxes
            self.buttons = Buttons()
            self.textBoxes = Boxes()
            
            yText = -1.5
            sizeText = 11
            
            # Textbox for residuals
            xpos = button_x0+3*(width+dwidth)
            self.axs.textBoxes.residuals = plt.axes([xpos, ypos, width, height])
            self.textBoxes.residuals = TextBox(self.axs.textBoxes.residuals, "", initial='0', color='lightgoldenrodyellow')
            self.axs.textBoxes.residuals.set_title('Residuals (d)', y=yText, size=sizeText)

            # Todo: Find out why the plot responds faster without this
            if self.extra_textBoxes:
                # Textbox for P0 
                xpos = button_x0+4*(width+dwidth)
                self.axs.textBoxes.P0 = self.fig.add_axes([xpos, ypos, width, height])
                self.textBoxes.P0 = TextBox(self.axs.textBoxes.P0, "", initial='0')
                self.connections_textBoxes.append(self.textBoxes.P0.on_submit(self.read_box_P0))
                self.axs.textBoxes.P0.set_title('$P_0$ (d)', y=yText, size=sizeText)
                # Textbox for dP0
                xpos = button_x0+5*(width+dwidth)
                self.axs.textBoxes.dP0 = self.fig.add_axes([xpos, ypos, width, height])
                self.textBoxes.dP0 = TextBox(self.axs.textBoxes.dP0, "", initial='0')
                self.connections_textBoxes.append(self.textBoxes.dP0.on_submit(self.read_box_dP0)) # TODO: Save connection id to later close it
                self.axs.textBoxes.dP0.set_title('$\Delta P_0$ (d)', y=yText, size=sizeText)
                # Textbox for Sigma
                xpos = button_x0+6*(width+dwidth)
                self.axs.textBoxes.Sigma = self.fig.add_axes([xpos, ypos, width, height])
                self.textBoxes.Sigma = TextBox(self.axs.textBoxes.Sigma, "", initial='0')
                self.connections_textBoxes.append(self.textBoxes.Sigma.on_submit(self.read_box_Sigma))
                self.axs.textBoxes.Sigma.set_title('$\Sigma$', y=yText, size=sizeText)
            
        # Create the figure and its axis
        fig_and_axs()
        # Create the sliders of the figure
        sliders()
        # Create the buttons of the figure
        buttons()
        
    def plot_p(self, pw=None):
        if pw is None:
            pw = self.pw
            
        # Clear plot if
        if self.plots.p_scatter_0:
            self.plots.p_scatter_0.remove()

        ax = self.axs.p
        x = pw.period.values
        y = np.repeat(0.2, x.size)
        color = self._colorOnOff[pw.selection.values]
        self.plots.p_scatter_0 = ax.scatter(x, y, c=color, marker=7, alpha=1, zorder=2, picker=5)

    def plot_pg(self):
        ax = self.axs.pg
        # Plot the periodogram of the light curve
        x = self.pg.period
        y = self.pg.ampl
        ax.plot(x, y, lw=1, color='k', zorder=3)
        # Mark level zero
        ax.axhline(0, ls='-', lw=0.5, color='gray') 
        
    def add_p2pg(self, pw=None):
        if pw is None:
            pw = self.pw

        # Clear plot if
        if self.plots.p_lines:            
            for line in self.plots.p_lines:
                line.remove()

        ax = self.axs.pg
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        p = pw.query('selection==1').period.values
        line = ax.plot(np.repeat(p, 3), np.tile([0, 1, np.nan], len(p)), lw=1, ls='dotted', color='k', transform=trans)
        self.plots.p_lines = line
        
    def plot_dp(self, pw=None):
        if pw is None:
            pw = self.pw
            
        # Clear plot if
        if self.plots.data2Dline_ax_pg:            
            for line in self.plots.data2Dline_ax_pg:
                line.remove()

        ax = self.axs.dp
        p = pw.query('selection==1').period.values
        x = period_for_dP_plot(p, mode='middle')
        y = np.diff(p)
        ls = self.ls_dp
        line = ax.plot(x, y, lw=1, color='k', ls=ls, marker='.', zorder=2, picker=5)
        self.plots.data2Dline_ax_pg = line
        # Mark level zero
        ax.axhline(0, ls='-', lw=0.5, color='gray') 

    def plot_echelle(self, keep_xlim=False, keep_ylim=False, keep_xlim_ylim=False, pw=None):
        if pw is None:
            pw = self.pw
            
        # Clear plot if
        if self.plots.echelle_vline1:
            self.plots.echelle_vline1.remove()
        if self.plots.echelle_vline2:
            self.plots.echelle_vline2.remove()
        if self.plots.p_scatter_1:
            self.plots.p_scatter_1.remove()
        if self.plots.p_scatter_2:
            self.plots.p_scatter_2.remove()
        if self.plots.p_scatter_3:
            self.plots.p_scatter_3.remove()
        
        ax = self.axs.echelle
        p = pw.period.values
        module_dp = self.module_dp
        ampl = pw.ampl.values
        selection = pw.selection.values
        color = self._colorOnOff[selection]
        size = 100.*(ampl/ampl.max())
        # Plotted range
        if keep_xlim_ylim:
            keep_xlim = True
            keep_ylim = True
        if keep_xlim:
            xlim = ax.get_xlim()
        if keep_ylim:
            ylim = ax.get_ylim()
        self.plots.p_scatter_1 = ax.scatter(p%module_dp-module_dp, p, s=size, color=color, zorder=3, picker=5)
        self.plots.p_scatter_2 = ax.scatter(p%module_dp+module_dp, p, s=size, color=color, zorder=3, picker=5)
        self.plots.p_scatter_3 = ax.scatter(p%module_dp, p, s=size, color=color, zorder=3, picker=5)
        ax.set_xlim(xlim) if keep_xlim else ax.set_xlim(-module_dp, 2*module_dp)
        ax.set_ylim(ylim) if keep_ylim else ax.set_ylim(p.min(), p.max())
            
        # Separe the 3 plotted echelles
        self.plots.echelle_vline1 = ax.axvline(0,  ls='dashed', color='gray', lw=2, zorder=2)
        self.plots.echelle_vline2 = ax.axvline(module_dp, ls='dashed', color='gray', lw=2, zorder=2)

    def plot_module_p(self):
        if self.plots.dp_hline:
            self.plots.dp_hline.remove()
        ax = self.axs.dp
        # line = ax_dp.axhline(dp, color='dodgerblue', lw=1, zorder=0, ls='dotted')
        self.plots.dp_hline = ax.axhline(self.module_dp, color='dodgerblue', lw=1, zorder=0, ls='-')
        

if __name__ == '__main__':

    # Read user inputs
    userData = UserData()
    
    # Generate the object to manage interactive plot
    extra_textBoxes = False
    iPlot = IPlot(pw=userData.pw_data,
                  pg=userData.pg_data,
                  freq_resolution=userData.pw_freq_resolution,
                  extra_textBoxes=extra_textBoxes,
                  TIC=userData.TIC)
    
    iPlot.connect()

    plt.show()
