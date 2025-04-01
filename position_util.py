import matplotlib.pyplot as plt
import neuropy.core as core
import numpy as np

def plot_avg_speed_over_position(
        position:core.Position,
        nbins = 100,
        index = [],
        color= "red",
        label='avg speed',
        ylabel = "Avg Speed (rad/s)",
        weights = 1, #if you want to weight speed
        ax=None
        ):
    """
    nbins: number of position bins to calculate avg speed over

    to use output: 
    _,ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(bin_centers, avg_speed, color='red', label="Avg Speed", lw=2)
    """

    # for now
    assert position.ndim == 1, "Currently only supports one dimensional position"

    if index== []:
        traces = position.traces[0]
        speed = position.speed
    else:
        traces = position.traces[0][index]
        speed = position.speed[index]

    xmin = min(traces)
    xmax = max(traces)
    bins = np.linspace(xmin, xmax, nbins+1)  # Creates nbins to calculate avg speed
    bin_centers = (bins[:-1] + bins[1:]) / 2# Use the midpoints of the bins for the x-values (positions)

    

    # Calculate average speed for each bin
    avg_speed = []
    for i in range(len(bins) - 1):
        in_pos_bin = (traces>bins[i]) & (traces<bins[i+1])
        bin_speeds = speed[in_pos_bin]
        if len(bin_speeds) > 0:  # Make sure there are values in this bin
            avg_speed.append(np.nanmean(bin_speeds))# Calculate the average speed for this bin
        else:
            avg_speed.append(np.nan)  # If no data in the bin, assign NaN

    avg_speed = avg_speed * weights # if you're plotting multiple things
    if weights == 1:
        label = label
    else:
        label = f'{label} * {weights}'

    if ax == None:
        _,ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(bin_centers, avg_speed, color=color, label=label, lw=2,zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Position')
    ax.legend()

    return 