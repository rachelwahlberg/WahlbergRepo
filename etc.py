def interpolate_position(position: core.Position,method='cubic'):
    """ to interpolate position for missing time points - removes all nans."""

    # Create an interpolation function

    t = np.arange(len(position.x))

    def fill_gaps(data, method):
        valid = ~np.isnan(data)  # Mask for valid (non-NaN) values
        if np.sum(valid) > 1:  # Ensure there are enough points to interpolate
            if method == 'linear':
                filled = np.copy(data)
                filled[np.isnan(data)] = np.interp(t[np.isnan(data)], t[valid], data[valid])
            else:  # Default to cubic
                interp_func = interp1d(
                    t[valid], data[valid], kind='cubic', bounds_error=False, fill_value="extrapolate"
                )
                filled = np.copy(data)
                filled[np.isnan(data)] = interp_func(t[np.isnan(data)])
            return filled
        else:
            return data  # Not enough points to interpolate; return as-is

    # Fill gaps for each dimension independently
    position.x = fill_gaps(position.x, method)
    position.y = fill_gaps(position.y, method)
    position.z = fill_gaps(position.z, method)

    return position



def remove_deadtimes(position: core.Position):
    """ manually inspect and cut dead time from position object. 
    readjusts start time if you cut off the beginning, otherwise it just nan's the cut out part. 
    to use, send in position object - will plot the traces over time, and you can specify start/end 
    (in sec from beginning) of time band you'd like to cut out. Replot after to confirm.
    """

#functional but using epochs is better.
#%matplotlib ipympl
#from neuropy.utils.position_util import remove_deadtimes
#pos = remove_deadtimes(pos)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    def remove_time_block(event):
            """ Removes the specified time block from the position data and closes the plot. """

            nonlocal position, fig, ax

            try:
                # Parse the input time block (e.g., "start_time,end_time")
                start_time, end_time = map(float, text_box.text.split(","))
                keep = (position.time < start_time) | (position.time > end_time)
                

                if keep[0]: #if you're removing something besides the beginning
                    updated_traces = position.traces
                    updated_traces[:,~keep] = np.nan
                else: #if you're removing the beginning, just cut off part instead of nan'ing
                    updated_traces = position.traces[:,keep]
                    
                    updated_time = position.time[keep]
                    position.t_start = updated_time[0] #only changes if you cut the beginning.

                #modify position
                position.traces = updated_traces
                
                
                plt.close(fig)
                print(f"Removed time block from {start_time} to {end_time}.")

            except ValueError:
                print("Invalid input format. Please enter in 'start_time,end_time' format.")

        # Plot x/y/z dimensions
    fig, ax = plt.subplots(3, 1, figsize=(12, 5), sharex=True)

    for a, dim, name in zip(ax, position.traces, ["x", "y", "z"]):
        a.plot(position.time, dim)
        a.set_ylabel(f"{name} (cm)")
    ax[0].set_xlabel("Time (s)")

    # Create a TextBox widget
    ax_box = plt.axes([0.5, 0, 0.3, 0.05])  # Position for the text box
    text_box = TextBox(ax_box, "Remove Time Block (sec) (start,end): ")
    plt.show()
    # Connect the TextBox to the function
    text_box.on_submit(remove_time_block)

    return position


    '''
    # in detect artifact epochs
    drops = np.zeros(zsc.shape)
    for i in range(1,len(zs_second)):
        if zs_second[i] == 0 and zs_second[i-1] == 0:
            drops[i] = 1
        else:
            drops[i] = 0

    import matplotlib.pyplot as plt
    xstarts = art_epochs.starts *1250
    xstops = art_epochs.stops * 1250
    y_min,y_max = plt.ylim()
    plt.plot(zsc)
    plt.vlines(xstarts,ymin=y_min,ymax=y_max,color="green",linewidth=1)
    plt.vlines(xstops,ymin=y_min,ymax=y_max,color="red",linewidth=1)
    plt.show()
    print(art_epochs.starts)
    print(art_epochs.stops)

'''
def decoder_pho(self,spkcount,ratemaps): #in decoders. doesn't work but didn't want to just delete.
    """
    ===========================
    Probability is calculated using this formula
    prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
    where,
        tau = binsize
    ===========================
    """
    # from scipy.special import factorial

    # tau = self.bin_size
    # nCells = spkcount.shape[0]
    # cell_prob = np.zeros((ratemaps.shape[1],spkcount.shape[1],nCells))
    # for cell in range(nCells):
    #     cell_spkcnt = spkcount[cell,:][np.newaxis,:]
    #     cell_ratemap = ratemaps[cell,:][:,np.newaxis]

    #     coeff = 1/(factorial(cell_spkcnt))
    #     # broadcasting
    #     cell_prob[:,:,cell]=(((tau*cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap))

    # posterior = np.prod(cell_prob,axis=2)
    # posterior /= np.sum(posterior,axis=0)

    # return posterior

    from scipy.special import gammaln

    ratemaps = np.clip(ratemaps, 1e-10, None)

    tau = self.bin_size
    nCells = spkcount.shape[0]
    cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))

    for cell in range(nCells):
        cell_spkcnt = spkcount[cell, :][np.newaxis, :]
        cell_ratemap = ratemaps[cell, :][:, np.newaxis]

        # Compute log likelihood to prevent overflow
        log_prob = (cell_spkcnt * np.log(tau * cell_ratemap) - tau * cell_ratemap - gammaln(cell_spkcnt + 1))
        cell_prob[:, :, cell] = np.exp(log_prob)

    # Combine across cells
    posterior = np.prod(cell_prob, axis=2)

    # Normalize posterior
    posterior /= (np.sum(posterior, axis=0) + 1e-10)

    return posterior
