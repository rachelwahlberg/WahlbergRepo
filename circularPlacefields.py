from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from copy import deepcopy
import seaborn as sns

from neuropy import core
from neuropy.utils.signal_process import ThetaParams
from neuropy import plotting
from neuropy.utils.mathutil import contiguous_regions
from neuropy.externals.peak_prominence2d import getProminence
from neuropy.analyses import Pf1D


class circularPF1D(Pf1D):
    def __init__(self, *args, **kwargs):
        """Inherits from neuropy.analyses.Pf1D to compute 1d place field using linearized coordinates,
        but adjusts for a circular track that is linearized in radians relative to center point of track.
        It always computes two place maps with and without speed thresholds.
        Parameters
        ----------
        neurons : core.Neurons
            neurons obj containing spiketrains and related info
        position: core.Position
            1D position
        grid_bin : int
            bin size of position bining, by default 5 cm
        epochs : core.Epoch,
            restrict calculation to these epochs, default None
        frate_thresh : float,
            peak firing rate should be above this value, default 1 Hz
        speed_thresh : float
            speed threshold for calculating place field, by default 3
        grid_bin: float, size of grid for calculating place field in cm, default 5
        sigma : float
            standard deviation for smoothing occupancy and spikecounts in each position bin, in units of cm, default 1 cm
        NOTE: speed_thresh is ignored if epochs is provided
        """
        #### overwrite default vals in Pf1D
        #kwargs.setdefault("speed_thresh",1)
        #kwargs.setdefault("grid_bin",0.5) #in radians
        super().__init__(*args,**kwargs)

        ######## TO DOS #######################################
        # check if speed_thresh is being used in anything 
        # check default vals of sigma, grid_bin for radian version, probs wrong.
        # go back and fix the speed caculation in linearize_position
        # import mask from 2D pf. have this as an import option 

    def plot_ratemap_by_time(self,ind=None,id=None,ax=None,timebin=10,**kwargs):
        """ can modify in plot_ratemap. those default args are:
        ax : [type], optional
            [description], by default None
        speed_thresh : bool, optional
            [description], by default False
        pad : int, optional
            [description], by default 2
        normalize_xbin : bool, optional
            [description], by default False
        normalize_tuning_curve : bool, optional
            [description], by default False
        cross_norm : np.array, optional
            Nx2 numpy array including xmin and xptp per neuron, by default None.
        sortby : array, optional
            [description], by default None
        cmap : str, optional
            [description], by default "tab20b"
        """

        #timebin in seconds.

        # Get neuron index
        assert (ind == None) != (id == None), "Exactly one of 'inds' and 'ids' must be a list or array"
        if ind is None:
            ind = np.where(id == self.neuron_ids)[0][0]

        # Slice desired neuron's placefield
        pfuse = self.neuron_slice([ind])

        # Get time slice info, convert into epochs

        # -- divide len by timebin to get # bins, round last bin up (so last bin is 10-20 sec rather than 0 - 10)
        epoch_starts = pfuse.time[0:timebin:-1]
        
        
        # nbins = 
        
        #set axes for fig

        pf_by_time = Pf1D()
        




        return plotting.plot_ratemap(self, **kwargs)
    



