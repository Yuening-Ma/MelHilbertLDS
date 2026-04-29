import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

# Disable matplotlib findfont warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set Times New Roman font
font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')

def get_times_font(size=12):
    """Get Times New Roman font with specified size"""
    return fm.FontProperties(fname=os.path.join(font_dir, 'times.ttf'), size=size)

def get_times_bold_font(size=12):
    """Get Times New Roman bold font with specified size"""
    return fm.FontProperties(fname=os.path.join(font_dir, 'timesbd.ttf'), size=size)

# Register font files to matplotlib
fm.fontManager.addfont(os.path.join(font_dir, 'times.ttf'))
fm.fontManager.addfont(os.path.join(font_dir, 'timesbd.ttf'))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

sys.path.append('gilbert/')
from gilbert2d import gilbert2d

def generate_hilbert_mapping(height, width):
    """Generate Hilbert curve mapping"""
    points = np.array(list(gilbert2d(width, height)))  # Note: width and height order
    mapping = np.zeros((height, width), dtype=int)
    for i, (x, y) in enumerate(points):
        mapping[y, x] = i
    return mapping

def generate_raster_mapping(height, width):
    """Generate raster scan mapping (C order)"""
    return np.arange(height*width).reshape(height, width)

def get_kernel_indices(mapping, center_i, center_j, kernel_size=3):
    """Get the original sequence indices covered by the convolution kernel centered at a specific position"""
    height, width = mapping.shape
    
    # Compute actual range of the convolution kernel
    start_i = max(0, center_i - kernel_size//2)
    end_i = min(height, center_i + kernel_size//2 + 1)
    start_j = max(0, center_j - kernel_size//2)
    end_j = min(width, center_j + kernel_size//2 + 1)
    
    # Get all indices covered by the convolution kernel region
    kernel_region = mapping[start_i:end_i, start_j:end_j]
    indices = kernel_region.flatten()
    
    return np.sort(indices)

def calculate_max_consecutive_length(indices):
    """Calculate the length of the longest consecutive sequence in the given index set"""
    if len(indices) == 0:
        return 0
    
    # Sort indices
    sorted_indices = np.sort(indices)
    
    # Calculate consecutive sequences
    max_consecutive_length = 1
    current_consecutive_length = 1
    
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            # Current number is consecutive with previous number
            current_consecutive_length += 1
        else:
            # Not consecutive, reset counter
            max_consecutive_length = max(max_consecutive_length, current_consecutive_length)
            current_consecutive_length = 1
    
    # Handle the last consecutive sequence
    max_consecutive_length = max(max_consecutive_length, current_consecutive_length)
    
    return max_consecutive_length

def visualize_mapping_with_ranges(mapping, kernel_size=3, title="", save_path=None):
    """Visualize the mapping and display the max consecutive sequence length value of the convolution kernel centered at each pixel"""
    height, width = mapping.shape
    consecutive_lengths = np.zeros_like(mapping, dtype=float)
    
    # Compute max consecutive sequence length value for each position
    for i in range(height):
        for j in range(width):
            indices = get_kernel_indices(mapping, i, j, kernel_size)
            consecutive_lengths[i, j] = calculate_max_consecutive_length(indices)
    
    # Create 1-row 2-column figure, set relative subplot widths
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1])
    
    # Left subplot: mapping and max consecutive sequence length values
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(mapping, cmap='viridis', aspect='equal')
    
    # Add max consecutive sequence length value on each pixel
    for i in range(height):
        for j in range(width):
            ax1.text(j, i, f'{consecutive_lengths[i,j]:.0f}', 
                    ha='center', va='center', color='white',
                    fontproperties=get_times_font(12))
    
    # ax1.set_title('Mapping with Max Consecutive Length Values')
    cbar = plt.colorbar(im, ax=ax1, label='Original Index')
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel('Original Index', fontproperties=get_times_font(14))
    
    # Right subplot: bar chart of max consecutive sequence length values
    ax2 = fig.add_subplot(gs[1])
    
    # Compute frequency of each value
    unique_values, counts = np.unique(consecutive_lengths.flatten(), return_counts=True)
    
    # Draw bar chart with specified color and uniform width
    bar_width = 0.6
    bars = ax2.bar(unique_values, counts, width=bar_width, color='gold', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add count value on top of each bar
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{int(count)}', 
                ha='center', va='bottom', fontproperties=get_times_bold_font())
    
    # ax2.set_title('Distribution of Max Consecutive Length Values')
    ax2.set_xlabel('Max Consecutive Length Value', fontproperties=get_times_font(14))
    ax2.set_ylabel('Frequency', fontproperties=get_times_font(14))
    
    # Set x-axis ticks to integers
    ax2.set_xticks(unique_values)
    ax2.set_xticklabels([int(x) for x in unique_values], fontproperties=get_times_font(14))
    
    # Add overall title
    # if title:
    #     fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return consecutive_lengths

def main():
    height = 8
    width = 16
    kernel_size = 3
    
    # Generate two mappings
    hilbert_map = generate_hilbert_mapping(height, width)
    raster_map = generate_raster_mapping(height, width)
    
    # Visualize and compute statistics
    hilbert_consecutive_lengths = visualize_mapping_with_ranges(
        hilbert_map, 
        kernel_size=kernel_size,
        title="Hilbert Curve Mapping (8x16)",
        save_path='code_for_manu/output/figure8a.png'
    )
    
    raster_consecutive_lengths = visualize_mapping_with_ranges(
        raster_map,
        kernel_size=kernel_size,
        title="Raster Scan Mapping (8x16)",
        save_path='code_for_manu/output/figure8b.png'
    )


if __name__ == "__main__":
    main() 