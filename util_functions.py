import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calculate_statistics(data, stat, confidence_interval):

    stat_values = data[stat].dropna()

    mean = np.mean(stat_values)
    std_dev = np.std(stat_values, ddof=1)  
    min_val = np.min(stat_values)
    max_val = np.max(stat_values)

    n = len(stat_values)
    if n > 1:
        se = std_dev / np.sqrt(n) 
        alpha = 1 - (confidence_interval / 100)
        z_score = np.abs(np.percentile(np.random.normal(0, 1, 1000000), [100 * (1 - alpha / 2)])[0])
        ci_range = z_score * se
        ci_lower = mean - ci_range
        ci_upper = mean + ci_range
    else:
        ci_lower = ci_upper = mean 

    return  mean, min_val , max_val,  std_dev, ci_lower, ci_upper


def plot_graph(df, stat, text_on_graph, player, text_type):
    # Create a smaller figure and axis object
    fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size
    df[stat] = df[stat].round(0).astype(int)

    # Plot the stat as a line with smaller markers and a softer blue
    ax.plot(df['WEEK'], df[stat], label=stat, color='#6699CC', marker='o', markersize=5)  # Softer blue (hex code)

    # Highlight points with text labels (smaller text size, black color)
    for i, row in df.iterrows():
        ax.text(row['WEEK'], row[stat] + text_on_graph, f"{int(round(row[stat]))}",
                ha='center', va='bottom', fontsize=8, color='black')  # Black text for the numbers

    # Calculate and plot the average line with a thinner line style
    avg = df[stat].mean()
    ax.axhline(y=avg, color='r', linestyle='--', linewidth=1, label="Average")  # Thinner line

    # Set axis labels and title with smaller fonts
    ax.set_xlabel(text_type, fontsize=9, labelpad=2)  # Reduced padding and font size
    ax.set_ylabel(stat, fontsize=9, labelpad=10)  # Reduced padding and font size
    ax.set_title(f"{player} {stat}", fontsize=9, fontweight='bold')

    # Dynamically adjust x-axis ticks using MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both", nbins=25))  # Max 6 ticks to avoid overcrowding

    # Customize y-axis ticks
    ax.tick_params(axis='y', labelsize=8)  # Smaller font size for y-axis ticks
    ax.tick_params(axis='x', labelsize=8)  # Smaller font size for x-axis ticks

    # Use tight_layout to avoid overlaps and ensure everything fits
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)