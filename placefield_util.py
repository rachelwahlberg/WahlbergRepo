import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from neuropy import core
from neuropy.utils import mathutil
from neuropy.plotting.figure import Fig

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import scipy.stats as stats
import seaborn as sns
import pandas as pd

def violin_plot(*datasets, column_name, perform_stat_tests=False, dataset_labels=None):
    """
    Create a violin plot comparing multiple datasets with statistical tests and connections between matching cell_ids.
    
    Parameters:
        datasets (list of pd.DataFrame): DataFrames containing the data to plot.
        column_name (str): The column to plot (e.g., "width_bin").
        perform_stat_tests (bool): Whether to perform statistical tests (default: False).
        dataset_labels (list of str): Labels for each dataset for legend and palette.
        colors (list of str): Colors for the violins.
    """

    #def lighten_color(color, amount=0.3):
    #    """
    #    Lighten a color by a specified amount.
    #    :param color: Color in any format accepted by matplotlib (e.g., 'blue', '#RRGGBB', etc.)
    #    :param amount: Amount to lighten the color (0 to 1)
    #    :return: A new color that is a lighter version of the input color.
    #    """
    #    c = mcolors.hex2color(mcolors.to_hex(color))  # Convert to RGB tuple
    #    return mcolors.to_hex([min(1, x + amount) for x in c])  # Lighten each channel

    datasets_peak0 = []

    #for i, data in enumerate(datasets):
    #    data = data[data['peak_no'] == 0].copy()  # Keep only rows where peak_no == 0
    #    data["Dataset"] = dataset_labels[i] if dataset_labels else f"Dataset {i+1}"  # Add Dataset column for plotting
    #    datasets_peak0.append(data)  # Append the modified data to the list

    # Concatenate all DataFrames into a single DataFrame
    #datasets_peak0 = pd.concat(datasets_peak0, ignore_index=True)

    #datasets_peak0 = []
    for i, data in enumerate(datasets):
        #data = data[data['peak_no'] == 0].deepcopy()  # Keep only rows where peak_no == 0
        data = data[(data['peak_no'] == 0) & (data[column_name].notna())].copy()
        data["Dataset"] = dataset_labels[i] if dataset_labels else f"Dataset {i+1}" #add Dataset col for plotting
        datasets_peak0.append(data) 

    # Combine all modified copies into a single DataFrame
    combined_data = []
    combined_data = pd.concat(datasets_peak0, ignore_index=True)
    
   # combined_data.dropna()
    #get different color per dataset for plotting
    def generate_colors(n):
        return sns.color_palette("husl", n)
    palette = generate_colors(len(datasets))

    # Plot the violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Dataset", y=column_name, data=combined_data, palette=palette, hue="Dataset", inner="box", legend=False)

    # Add lines and scatter points between matching cell_ids
    common_ids = pd.merge(datasets_peak0[0][['cell_id']], datasets_peak0[1][['cell_id']], on="cell_id", how="inner")

    for cell_id in common_ids['cell_id']:
        # Get the matching rows for peak_no == 0 in each dataset
        rows = []
        for data in datasets_peak0:
            row = data[(data['cell_id'] == cell_id) & (data['peak_no'] == 0)]
            rows.append(row)
        
        # Add line connecting matching points
        plt.plot(
            [0, 1],  # x positions of the two violins (0 for first dataset, 1 for second dataset)
            [rows[0][column_name].values[0], rows[1][column_name].values[0]],  # y positions (column_name is provided)
            color='black',  # line color
            lw=1  # line width
        )

        # Add orange points at the ends of the lines
        plt.scatter(
            [0, 1],  # x positions for the points (0 for first dataset, 1 for second dataset)
            [rows[0][column_name].values[0], rows[1][column_name].values[0]],  # y positions for the points
            color='black',  # color of the points
            zorder=5,  # make sure the points are above the lines
            label="Matching Points" if cell_id == common_ids['cell_id'].iloc[0] else ""  # Label only for the first match
        )

    # Plot unmatched cell_ids as dots with a different color
    unmatched_data = []
    for i, data in enumerate(datasets_peak0):
        unmatched_data.append(data[~data['cell_id'].isin(common_ids['cell_id'])])

    # Unmatched points from each dataset (customized colors for each dataset)
    for i, data in enumerate(unmatched_data):
        label = f"Unmatched" if i == 0 else None

        plt.scatter(
            data["Dataset"].map(lambda x: i),  # x position for the current dataset
            data[column_name],  # y position for the specified column
            color='gray',  # color for unmatched points
            label=label,
            marker="o",  # dot marker
            zorder=5  # make sure dots are on top of the violins
        )

    # Labels & title
    plt.xlabel("Dataset")
    plt.ylabel(column_name)
    plt.title(f"Comparison of {column_name} Distributions", fontsize=16, y=1.05)

    # Display the legend
    plt.legend()

        # Extend the y-axis slightly to accommodate the bracket above the violins
    current_ylim = plt.gca().get_ylim()  # Get current y-axis limits
    y_max = max(combined_data[column_name].max(), current_ylim[1])  # Get the maximum value and extend if needed
    plt.ylim(0, y_max + 0.2)  # Add extra space above the violins for the bracket


    if perform_stat_tests:

        significance_threshold = 0.05
        # Perform t-tests for each pair of datasets
        num_datasets = len(datasets_peak0)
        for i in range(num_datasets):
            for j in range(i+1, num_datasets):
                # Perform t-test between datasets[i] and datasets[j]
                t_stat, p_value = stats.ttest_ind(datasets_peak0[i][column_name], datasets_peak0[j][column_name], equal_var=False)

                # Plot a horizontal bracket line if significant
                if p_value < significance_threshold:
                    # Define bracket height and offset
                    bracket_height = 0.1
                    offset = 0.02

                    # Plot the horizontal bracket (a line with vertical ticks on both ends)
                    plt.plot([i, i], [y_max + bracket_height, y_max + offset], color='black', lw=2)  # left tick for i
                    plt.plot([j, j], [y_max + bracket_height, y_max + offset], color='black', lw=2)  # right tick for j
                    plt.plot([i, j], [y_max + bracket_height, y_max + bracket_height], color='black', lw=2)  # horizontal line

                    # Annotate with a star and p-value
                    plt.text((i + j) / 2, y_max + bracket_height + 0.02, f"* p={p_value:.3f}", ha="center", va="bottom", fontsize=12)

    # Show the plot
    plt.show()

def violin_plot2(*datasets:pd.DataFrame, column_name, perform_stat_tests=False, dataset_labels=None):
    """
    Create a violin plot comparing multiple datasets with statistical tests and connections between matching cell_ids.
    
    Parameters:
        datasets (list of pd.DataFrame): DataFrames containing the data to plot.
        column_name (str): The column to plot (e.g., "width_bin").
        perform_stat_tests (bool): Whether to perform statistical tests (default: False).
        dataset_labels (list of str): Labels for each dataset for legend and palette.
        colors (list of str): Colors for the violins.
    """

    #def lighten_color(color, amount=0.3):
    #    """
    #    Lighten a color by a specified amount.
    #    :param color: Color in any format accepted by matplotlib (e.g., 'blue', '#RRGGBB', etc.)
    #    :param amount: Amount to lighten the color (0 to 1)
    #    :return: A new color that is a lighter version of the input color.
    #    """
    #    c = mcolors.hex2color(mcolors.to_hex(color))  # Convert to RGB tuple
    #    return mcolors.to_hex([min(1, x + amount) for x in c])  # Lighten each channel
    
    # Create a new column for dataset labels and copy the datasets to avoid modifying the originals
    

    # Filter both datasets to include only rows where peak_no == 0
    #pf1cl_data = pf1cl_data[pf1cl_data['peak_no'] == 0]
    #pf2cl_data = pf2cl_data[pf2cl_data['peak_no'] == 0]

    #datasets_copies = [data.copy() for data in datasets]  # Create copies first

    for i, data in enumerate(datasets):
        data = temp_data[temp_data['peak_no'] == 0]  # Keep only rows where peak_no == 0
        temp_data["Dataset"] = dataset_labels[i] if dataset_labels else f"Dataset {i+1}"
        datasets_copies[i] = temp_data  # Store the filtered data back

    # Combine all modified copies into a single DataFrame
    combined_data = []
    combined_data = pd.concat(datasets_copies, ignore_index=True)
    
    def generate_colors(n):
        return sns.color_palette("husl", n)
    palette = generate_colors(len(datasets))

    # Plot the violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Dataset", y=column_name, data=combined_data, palette=palette, hue="Dataset", inner="box", legend=False)

    # Add lines and scatter points between matching cell_ids
    common_ids = pd.merge(datasets[0][['cell_id']], datasets[1][['cell_id']], on="cell_id", how="inner")

    for cell_id in common_ids['cell_id']:
        # Get the matching rows for peak_no == 0 in each dataset
        rows = []
        for data in datasets_copies:
            row = data[(data['cell_id'] == cell_id) & (data['peak_no'] == 0)]
            rows.append(row)
        
        # Add line connecting matching points
        plt.plot(
            [0, 1],  # x positions of the two violins (0 for first dataset, 1 for second dataset)
            [rows[0][column_name].values[0], rows[1][column_name].values[0]],  # y positions (column_name is provided)
            color='black',  # line color
            lw=1  # line width
        )

        # Add orange points at the ends of the lines
        plt.scatter(
            [0, 1],  # x positions for the points (0 for first dataset, 1 for second dataset)
            [rows[0][column_name].values[0], rows[1][column_name].values[0]],  # y positions for the points
            color='black',  # color of the points
            zorder=5,  # make sure the points are above the lines
            label="Matching Points" if cell_id == common_ids['cell_id'].iloc[0] else ""  # Label only for the first match
        )

    # Plot unmatched cell_ids as dots with a different color
    unmatched_data = []
    for i, data in enumerate(datasets_copies):
        unmatched_data.append(data[~data['cell_id'].isin(common_ids['cell_id'])])

    # Unmatched points from each dataset (customized colors for each dataset)
    for i, data in enumerate(unmatched_data):
        label = f"Unmatched" if i == 0 else None

        plt.scatter(
            data["Dataset"].map(lambda x: i),  # x position for the current dataset
            data[column_name],  # y position for the specified column
            color='gray',  # color for unmatched points
            label=label,
            marker="o",  # dot marker
            zorder=5  # make sure dots are on top of the violins
        )

    # Labels & title
    plt.xlabel("Dataset")
    plt.ylabel(column_name)
    plt.title(f"Comparison of {column_name} Distributions", fontsize=16, y=1.05)

    # Display the legend
    plt.legend()

        # Extend the y-axis slightly to accommodate the bracket above the violins
    current_ylim = plt.gca().get_ylim()  # Get current y-axis limits
    y_max = max(combined_data[column_name].max(), current_ylim[1])  # Get the maximum value and extend if needed
    plt.ylim(0, y_max + 0.2)  # Add extra space above the violins for the bracket


    if perform_stat_tests:

        significance_threshold = 0.05
        # Perform t-tests for each pair of datasets
        num_datasets = len(datasets)
        for i in range(num_datasets):
            for j in range(i+1, num_datasets):
                # Perform t-test between datasets[i] and datasets[j]
                t_stat, p_value = stats.ttest_ind(datasets[i][column_name], datasets[j][column_name], equal_var=False)

                # Plot a horizontal bracket line if significant
                if p_value < significance_threshold:
                    # Define bracket height and offset
                    bracket_height = 0.1
                    offset = 0.02

                    # Plot the horizontal bracket (a line with vertical ticks on both ends)
                    plt.plot([i, i], [y_max + bracket_height, y_max + offset], color='black', lw=2)  # left tick for i
                    plt.plot([j, j], [y_max + bracket_height, y_max + offset], color='black', lw=2)  # right tick for j
                    plt.plot([i, j], [y_max + bracket_height, y_max + bracket_height], color='black', lw=2)  # horizontal line

                    # Annotate with a star and p-value
                    plt.text((i + j) / 2, y_max + bracket_height + 0.02, f"* p={p_value:.3f}", ha="center", va="bottom", fontsize=12)

    # Show the plot
    plt.show()

def plot_ratemap_dual(    #NOT CURRENTLY WORKING
    ratemap1,  # First Ratemap (plotted to the left)
    ratemap2,  # Second Ratemap (plotted to the right)
    normalize_xbin=False,
    ax=None,
    pad=2,
    normalize_tuning_curve=False,
    cross_norm=None,
    sortby=None,
    cmap="tab20b",
):
    """
    Most code is from ratemaps.plot_ratemaps, but mirrors two ratemaps
    on a shared y-axis. Good for observing differences between two conditions.
    
    - `ratemap1` goes **left** (negative x-axis)
    - `ratemap2` goes **right** (positive x-axis)
    - Y-axis (position) remains shared
    """
    cmap = mpl.cm.get_cmap(cmap)

    tuning_curves1 = ratemap1.tuning_curves
    tuning_curves2 = ratemap2.tuning_curves
    n_neurons = ratemap1.n_neurons

    bin_cntr = ratemap1.x_coords()  # Common position bins

    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))

    if normalize_tuning_curve:
        tuning_curves1 = mathutil.min_max_scaler(tuning_curves1)
        tuning_curves2 = mathutil.min_max_scaler(tuning_curves2)
        pad = 1  # Reduce spacing for better visualization

    if sortby is None:
        sort_ind = np.argsort(np.argmax(tuning_curves1, axis=1))
    else:
        sort_ind = sortby

    for i, neuron_ind in enumerate(sort_ind):
        color = cmap(i / len(sort_ind))

        # Left (Negative) - Ratemap 1
        ax.fill_between(
            -tuning_curves1[neuron_ind], bin_cntr,  # Flip x-direction
            bin_cntr, i * pad,
            color=color, alpha=0.7, zorder=i + 1,
        )
        ax.plot(-tuning_curves1[neuron_ind], bin_cntr, color=color, alpha=1, lw=0.6)

        # Right (Positive) - Ratemap 2
        ax.fill_between(
            tuning_curves2[neuron_ind], bin_cntr,
            bin_cntr, i * pad,
            color=color, alpha=0.7, zorder=i + 1,
        )
        ax.plot(tuning_curves2[neuron_ind], bin_cntr, color=color, alpha=1, lw=0.6)

    # Formatting
    ax.set_xlabel("Firing Rate (Left: Ratemap 1 | Right: Ratemap 2)")
    ax.set_ylabel("Position")
    ax.axvline(0, color="black", linewidth=1)  # Zero reference line
    ax.spines["left"].set_visible(False)
    ax.set_yticks(list(range(len(sort_ind))))
    ax.set_yticklabels(list(ratemap1.neuron_ids[sort_ind]))
    ax.set_xlim([-np.max(tuning_curves1), np.max(tuning_curves2)])  # Symmetric x-axis
    ax.invert_yaxis()  # Maintain position ordering

    return ax
