import matplotlib.pyplot as plt
import numpy as np

def create_probability_line_plot(alpha_1: float, alpha_2: float):

    assert 0 <= alpha_1 <= 1, "alpha_1 must be between 0 and 1"
    assert 0 <= alpha_2 <= 1, "alpha_2 must be between 0 and 1"
    assert alpha_1 < alpha_2, "alpha_1 must be smaller than alpha_2"

    fig, ax = plt.subplots(figsize=(10, 2)) # Adjust figsize for desired width and height

    # --- 1. The main number line ---
    # Draw a thicker, underlying line for context
    ax.hlines(y=0, xmin=0, xmax=1, color='white', linewidth=2)

    # --- 2. Ticks and Labels ---
    tick_positions = [0, alpha_1, alpha_2, 1]
    
    # Add minor ticks for better visual using vlines
    minor_ticks = np.linspace(0, 1, 11) # 0.1 intervals
    # CORRECTED LINE: Use vlines to draw short vertical ticks
    ax.vlines(x=minor_ticks, ymin=-0.01, ymax=0.01, color='white', linewidth=1, alpha=0.5)

    # Add the main ticks
    for pos in tick_positions:
        ax.vlines(x=pos, ymin=-0.02, ymax=0.02, color='white', linewidth=2) # Main ticks

    # Add numeric labels below the line
    ax.text(0, -0.05, '0', color='white', ha='center', va='top', fontsize=12)
    ax.text(alpha_1, -0.05, str(alpha_1), color='white', ha='center', va='top', fontsize=12)
    ax.text(alpha_2, -0.05, str(alpha_2), color='white', ha='center', va='top', fontsize=12)
    ax.text(1, -0.05, '1', color='white', ha='center', va='top', fontsize=12)

    # Add mathematical expressions below the line
    ax.text(alpha_1, -0.15, r'$1 - \max(\alpha_1, \alpha_2)$', color='white', ha='center', va='top', fontsize=12)
    ax.text(alpha_2, -0.15, r'$1 - \alpha_1$', color='white', ha='center', va='top', fontsize=12)

    # --- 3. Shaded Regions/Bars ---
    bar_height = 0.1 # Height of the shaded bars
    bar_y_position = 0.05 # Slightly above the line
    bar_color = '#642f00' # Dark brown color

    # Region 1: from 0 to alpha_1
    ax.add_patch(plt.Rectangle((0, bar_y_position), alpha_1, bar_height,
                               facecolor=bar_color, edgecolor='none', zorder=0))
    ax.text(alpha_1 / 2, bar_y_position + bar_height / 2, r'$\emptyset$',
            color='white', ha='center', va='center', fontsize=20, weight='bold')

    # Region 2: from alpha_2 to 1
    ax.add_patch(plt.Rectangle((alpha_2, bar_y_position), 1 - alpha_2, bar_height,
                               facecolor=bar_color, edgecolor='none', zorder=0))
    ax.text((alpha_2 + 1) / 2, bar_y_position + bar_height / 2, r'$\{\bullet, \circ\}$',
            color='white', ha='center', va='center', fontsize=16)


    # --- 4. Set Notations above the line ---
    text_y_position = bar_y_position + bar_height + 0.03 # Position above the bars
    
    # Set at midpoint between alpha_1 and alpha_2)
    ax.text((alpha_1 + alpha_2) / 2, bar_y_position + bar_height / 2, r'$\{\bullet\}$',
        color='white', ha='center', va='center', fontsize=16)

    # --- 5. Styling and Removing unnecessary plot elements ---
    ax.set_xlim(-0.05, 1.05) # Slightly extend x-axis for padding
    ax.set_ylim(-0.25, 0.25) # Adjust y-axis limits to fit all text and bars

    ax.axis('off') # Hide the default Matplotlib axes (spines, ticks, labels)
    ax.set_facecolor('black') # Set the background color of the plot area

    # Set the figure facecolor to black
    fig.patch.set_facecolor('black')

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig("prova2.png")

# Call the function to create the plot
create_probability_line_plot(0.68, 0.92)
