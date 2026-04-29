import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pandas as pd

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

def lds_sampling(original_length, target_length, seed=None):
    """
    Downsample using Low Discrepancy Sequence (LDS), randomly select starting point

    Args:
        original_length (int): Original sequence length
        target_length (int): Target sequence length
        seed (int): Random seed for selecting starting point

    Returns:
        numpy.ndarray: Selected index array
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate complete LDS sequence
    a = np.arange(1, original_length+1) * np.e % 1.0
    r = np.argsort(a)
    
    # Randomly select starting point
    max_start = len(r) - target_length
    start_index = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    
    # Select specified number of points from random starting point
    selected_indices = sorted(r[start_index:start_index+target_length])
    return np.array(selected_indices)

def uniform_sampling(original_length, target_length, start_index=0):
    """
    Uniform sampling downsampling

    Args:
        original_length (int): Original sequence length
        target_length (int): Target sequence length
        start_index (int): Starting index offset

    Returns:
        numpy.ndarray: Selected index array
    """
    step = original_length / target_length
    indices = np.arange(start_index, original_length, step, dtype=np.int32)[:target_length]
    return indices

def random_sampling(original_length, target_length, seed=None):
    """
    Random sampling downsampling

    Args:
        original_length (int): Original sequence length
        target_length (int): Target sequence length
        seed (int): Random seed

    Returns:
        numpy.ndarray: Selected index array
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.sort(np.random.choice(original_length, target_length, replace=False))
    return indices

def generate_signal_type_a(N_L, num_pulses=3, noise_std=0.1, seed=None):
    """
    Generate type A signal: stationary signal with transient pulses

    Args:
        N_L (int): Sequence length
        num_pulses (int): Number of pulses
        noise_std (float): Noise standard deviation
        seed (int): Random seed

    Returns:
        numpy.ndarray: Generated signal
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate background noise
    signal = np.random.normal(0, noise_std, N_L)
    
    # Add pulses
    for _ in range(num_pulses):
        pos = np.random.randint(0, N_L)
        width = np.random.randint(5, 20)
        amplitude = np.random.uniform(0.5, 2.0)
        
        # Generate Gaussian pulse
        t = np.arange(-width, width)
        gaussian_pulse = amplitude * np.exp(-0.5 * (t/width*3)**2)
        
        # Add pulse to signal
        start_idx = max(0, pos - width)
        end_idx = min(N_L, pos + width)
        pulse_start = max(0, width - pos)
        pulse_end = min(2*width, width + (N_L - pos))
        signal[start_idx:end_idx] += gaussian_pulse[pulse_start:pulse_end]
    
    return signal

def generate_signal_type_b(N_L, num_components=3, noise_std=0.1, seed=None):
    """
    Generate type B signal: signal with periodic components

    Args:
        N_L (int): Sequence length
        num_components (int): Number of sine wave components
        noise_std (float): Noise standard deviation
        seed (int): Random seed

    Returns:
        numpy.ndarray: Generated signal
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(N_L)
    signal = np.zeros(N_L)
    
    # Add multiple sine wave components
    for _ in range(num_components):
        freq = np.random.uniform(1, 10)  # Frequency range
        phase = np.random.uniform(0, 2*np.pi)  # Random phase
        amplitude = np.random.uniform(0.2, 1.0)  # Random amplitude
        signal += amplitude * np.sin(2*np.pi*freq*t/N_L + phase)
    
    # Add noise
    signal += np.random.normal(0, noise_std, N_L)
    
    return signal

def generate_signal_type_c(N_L, noise_std=0.1, seed=None):
    """
    Generate type C signal: slowly varying trend signal

    Args:
        N_L (int): Sequence length
        noise_std (float): Noise standard deviation
        seed (int): Random seed

    Returns:
        numpy.ndarray: Generated signal
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(N_L)
    
    # Generate base trend
    trend = 0.5 * np.sin(2*np.pi*t/N_L) + 0.3 * np.sin(4*np.pi*t/N_L)
    
    # Add high-frequency small fluctuations
    high_freq = 0.1 * np.sin(20*np.pi*t/N_L)
    
    # Combine signal and add noise
    signal = trend + high_freq + np.random.normal(0, noise_std, N_L)
    
    return signal

def analyze_sampling_uniformity(indices, N_L, title="Sampling Point Distribution"):
    """
    Analyze sampling point uniformity

    Args:
        indices (numpy.ndarray): Sampling indices
        N_L (int): Original sequence length
        title (str): Chart title

    Returns:
        tuple: (mean gap, std gap)
    """
    # Compute adjacent index gaps
    gaps = np.diff(indices)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    
    # # Create chart
    # plt.figure(figsize=(12, 4))
    
    # # Plot sampling point distribution
    # plt.subplot(121)
    # plt.plot(indices, np.zeros_like(indices), 'o', label='Sampling Points')
    # plt.xlim(0, N_L)
    # plt.title(f"{title}\nMean Gap: {mean_gap:.2f}, Std: {std_gap:.2f}")
    # plt.xlabel('Time Index')
    # plt.grid(True)
    
    # # Plot gap distribution histogram
    # plt.subplot(122)
    # plt.hist(gaps, bins=20, density=True, alpha=0.7)
    # plt.axvline(mean_gap, color='r', linestyle='--', label=f'Mean Gap: {mean_gap:.2f}')
    # plt.title('Gap Distribution')
    # plt.xlabel('Gap')
    # plt.ylabel('Density')
    # plt.grid(True)
    # plt.legend()
    
    # plt.tight_layout()
    
    return mean_gap, std_gap

def evaluate_signal_preservation(original_signal, sampled_indices, method_name=""):
    """
    Evaluate signal preservation capability

    Args:
        original_signal (numpy.ndarray): Original signal
        sampled_indices (numpy.ndarray): Sampling indices
        method_name (str): Method name

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Get sampled signal
    sampled_signal = original_signal[sampled_indices]
    
    # Linear interpolation back to original length
    x_new = np.arange(len(original_signal))
    f = interp1d(sampled_indices, sampled_signal, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolated_signal = f(x_new)
    
    # Compute evaluation metrics
    mse = mean_squared_error(original_signal, interpolated_signal)
    correlation = np.corrcoef(original_signal, interpolated_signal)[0, 1]
    
    # Compute spectrum (for plotting)
    original_spectrum = np.abs(fft(original_signal))
    interpolated_spectrum = np.abs(fft(interpolated_signal))
    
    return {
        'MSE': mse,
        'Correlation': correlation,
        'Interpolated_Signal': interpolated_signal,
        'Original_Spectrum': original_spectrum,
        'Interpolated_Spectrum': interpolated_spectrum
    }

def plot_signal_comparison(original_signal, sampling_results, signal_type):
    """
    Plot signal comparison chart

    Args:
        original_signal (numpy.ndarray): Original signal
        sampling_results (dict): Results from different sampling methods
        signal_type (str): Signal type description
    """
    plt.figure(figsize=(14, 6))
    
    # Plot original signal
    plt.subplot(2, 2, 1)
    t = np.arange(len(original_signal))
    plt.plot(t, original_signal, 'k-', label='Original Signal', linewidth=1)
    plt.title(f'Type {signal_type} Original Signal', fontproperties=get_times_bold_font(16))
    plt.xlabel('Time', fontproperties=get_times_font(14))
    plt.ylabel('Amplitude', fontproperties=get_times_font(14))
    plt.grid(True)
    plt.legend(prop=get_times_font())
    
    # Plot reconstruction results for various methods
    colors = ['b', 'r', 'g']
    for (method_name, result), color, subplot_idx in zip(sampling_results.items(), colors, [2, 3, 4]):
        plt.subplot(2, 2, subplot_idx)
        
        # Plot original signal as reference
        plt.plot(t, original_signal, 'k-', label='Original', alpha=0.3, linewidth=1)
        
        # Plot reconstructed signal and sampling points
        plt.plot(t, result['Interpolated_Signal'], f'{color}-', 
                label=f'Reconstructed', alpha=0.7, linewidth=1)
        sampled_indices = result['Indices']
        plt.plot(sampled_indices, original_signal[sampled_indices], 
                f'{color}o', label=f'Sampled Points', alpha=0.7, markersize=2)
        
        plt.title(f'{method_name} Reconstruction\nMSE: {result["MSE"]:.6f}, Correlation: {result["Correlation"]:.3f}', fontproperties=get_times_bold_font(16))
        plt.xlabel('Time', fontproperties=get_times_font(14))
        plt.ylabel('Amplitude', fontproperties=get_times_font(14))
        plt.grid(True)
        plt.legend(prop=get_times_font())
    
    plt.tight_layout()

def run_experiment(signal_type='A', N_L=1024, N_L_target=256, seeds=[7, 42, 1309]):
    """
    Run complete experiment

    Args:
        signal_type (str): Signal type ('A', 'B', or 'C')
        N_L (int): Original signal length
        N_L_target (int): Target signal length
        seeds (list): List of random seeds to use
    """
    # Generate signal
    signal_generators = {
        'A': generate_signal_type_a,
        'B': generate_signal_type_b,
        'C': generate_signal_type_c
    }
    
    # Store results for all seeds
    all_results = {seed: {} for seed in seeds}
    
    # Generate signal for first seed (for display)
    first_seed = seeds[0]
    signal_generator = signal_generators[signal_type]
    display_signal = signal_generator(N_L, seed=first_seed)
    
    for seed in seeds:
        # Generate new signal for each seed
        original_signal = signal_generator(N_L, seed=seed)
        
        # Apply different sampling methods
        sampling_methods = {
            'LDS': lambda: lds_sampling(N_L, N_L_target, seed=seed),
            'UDS': lambda: uniform_sampling(N_L, N_L_target),
            'RS': lambda: random_sampling(N_L, N_L_target, seed=seed)
        }
        
        # Sample once for each method
        for method_name, sampling_func in sampling_methods.items():
            # Get sampling indices
            indices = sampling_func()
            
            # Evaluate signal preservation capability
            result = evaluate_signal_preservation(original_signal, indices, method_name)
            result['Indices'] = indices
            all_results[seed][method_name] = result
    
    # Compute average results (only average evaluation metrics)
    avg_results = {}
    for method in ['LDS', 'UDS', 'RS']:
        # Use first seed's reconstruction result
        first_result = all_results[first_seed][method]
        avg_results[method] = {
            'MSE': 0,
            'Correlation': 0,
            'Interpolated_Signal': first_result['Interpolated_Signal'],
            'Indices': first_result['Indices']
        }
        
        # Only average evaluation metrics
        for seed in seeds:
            avg_results[method]['MSE'] += all_results[seed][method]['MSE']
            avg_results[method]['Correlation'] += all_results[seed][method]['Correlation']
        
        # Compute average of evaluation metrics
        n_seeds = len(seeds)
        avg_results[method]['MSE'] /= n_seeds
        avg_results[method]['Correlation'] /= n_seeds
    
    # Plot signal comparison (using first seed's reconstruction result)
    plot_signal_comparison(display_signal, avg_results, signal_type)
    
    # Save signal preservation evaluation results
    preservation_data = []
    metrics = ['MSE', 'Correlation']
    for method in ['LDS', 'UDS', 'RS']:
        row = {'Method': method}
        for metric in metrics:
            if len(seeds) == 1:
                # If only one seed, use metric name directly as column name
                row[metric] = all_results[seeds[0]][method][metric]
            else:
                # If multiple seeds, add each seed's result and average
                for seed in seeds:
                    row[f'{metric}_seed_{seed}'] = all_results[seed][method][metric]
                row[f'{metric}_avg'] = avg_results[method][metric]
        preservation_data.append(row)
    
    preservation_df = pd.DataFrame(preservation_data)
    preservation_df.to_csv(f'code_for_manu/output/figure6_{signal_type}.csv', index=False)

# Run experiment
if __name__ == "__main__":
    # Run experiments for different signal types
    for signal_type in ['A', 'B', 'C']:
        run_experiment(signal_type=signal_type, seeds=[7,42,123,1309,5287,31415])
        plt.savefig(f'code_for_manu/output/figure6_type_{signal_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
