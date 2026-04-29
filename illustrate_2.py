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
import librosa
import librosa.display
from skimage.metrics import structural_similarity as ssim

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
    # print('original_length', original_length, 'target_length', target_length, 'step', step)
    indices = np.arange(start_index, original_length, step)[:target_length].astype(np.int32)
    # print('indices', indices)
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
    
    # Prepare 1D arrays for DTW
    original_1d = original_signal.reshape(-1, 1)
    sampled_1d = sampled_signal.reshape(-1, 1)
    dtw_distance, _ = fastdtw(original_1d, sampled_1d, dist=euclidean)
    
    # Compute spectrum
    original_spectrum = np.abs(fft(original_signal))
    interpolated_spectrum = np.abs(fft(interpolated_signal))
    
    # Compute spectral KL divergence
    original_spectrum_norm = original_spectrum / np.sum(original_spectrum)
    interpolated_spectrum_norm = interpolated_spectrum / np.sum(interpolated_spectrum)
    kl_divergence = np.sum(np.where(original_spectrum_norm > 0,
                                   original_spectrum_norm * np.log(original_spectrum_norm / (interpolated_spectrum_norm + 1e-10)),
                                   0))
    
    # Compute spectral Euclidean distance
    spectral_distance = np.sqrt(np.sum((original_spectrum - interpolated_spectrum)**2))
    
    return {
        'MSE': mse,
        'Correlation': correlation,
        'DTW': dtw_distance,
        'KL_Divergence': kl_divergence,
        'Spectral_Distance': spectral_distance,
        'Interpolated_Signal': interpolated_signal,
        'Original_Spectrum': original_spectrum,
        'Interpolated_Spectrum': interpolated_spectrum
    }

def plot_signal_comparison(original_signal, sampling_results, signal_type=""):
    """
    Plot signal comparison chart

    Args:
        original_signal (numpy.ndarray): Original signal
        sampling_results (dict): Results from different sampling methods
        signal_type (str): Signal type description
    """
    plt.figure(figsize=(15, 10))
    
    # Plot time-domain comparison
    plt.subplot(211)
    t = np.arange(len(original_signal))
    plt.plot(t, original_signal, 'k-', label='Original Signal', alpha=0.5)
    
    colors = ['b', 'r', 'g']
    for (method_name, result), color in zip(sampling_results.items(), colors):
        plt.plot(t, result['Interpolated_Signal'], f'{color}-', 
                label=f'{method_name} Reconstructed', alpha=0.7, linewidth=1, markersize=5)
        sampled_indices = result['Indices']
        plt.plot(sampled_indices, original_signal[sampled_indices], 
                f'{color}o', label=f'{method_name} Points', alpha=0.7, linewidth=1, markersize=5)
    
    plt.title(f'Type {signal_type} Signal Reconstruction Comparison', fontproperties=get_times_bold_font(16))
    plt.xlabel('Time', fontproperties=get_times_font(16))
    plt.ylabel('Amplitude', fontproperties=get_times_font(16))
    plt.grid(True)
    plt.legend(prop=get_times_font())
    
    # Plot spectrum comparison
    plt.subplot(212)
    freqs = fftfreq(len(original_signal))
    plt.plot(freqs[:len(freqs)//2], np.abs(sampling_results['LDS']['Original_Spectrum'])[:len(freqs)//2], 
             'k-', label='Original Spectrum', alpha=0.5, linewidth=1)
    
    for (method_name, result), color in zip(sampling_results.items(), colors):
        plt.plot(freqs[:len(freqs)//2], np.abs(result['Interpolated_Spectrum'])[:len(freqs)//2], 
                f'{color}-', label=f'{method_name} Spectrum', alpha=0.7, linewidth=1)
    
    plt.title('Spectrum Comparison', fontproperties=get_times_bold_font(16))
    plt.xlabel('Frequency', fontproperties=get_times_font(16))
    plt.ylabel('Magnitude', fontproperties=get_times_font(16))
    plt.grid(True)
    plt.legend(prop=get_times_font())
    
    plt.tight_layout()

def generate_mel_spectrogram(signal_type='A', N_L=22050, sr=22050, n_fft=1024, hop_length=64, seed=None):
    """
    Generate Mel spectrogram

    Args:
        signal_type (str): Signal type ('A', 'B', or 'C')
        N_L (int): Sequence length, default is 1-second audio length
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Hop length
        seed (int): Random seed

    Returns:
        tuple: (Mel spectrogram, number of time frames)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate different types of signals
    if signal_type == 'A':
        # Stationary signal with transient pulses
        signal = np.random.normal(0, 0.1, N_L)
        for _ in range(3):
            pos = np.random.randint(0, N_L)
            width = np.random.randint(100, 400)  # Increase pulse width for longer signals
            amplitude = np.random.uniform(0.5, 2.0)
            t = np.arange(-width, width)
            gaussian_pulse = amplitude * np.exp(-0.5 * (t/width*3)**2)
            start_idx = max(0, pos - width)
            end_idx = min(N_L, pos + width)
            pulse_start = max(0, width - pos)
            pulse_end = min(2*width, width + (N_L - pos))
            signal[start_idx:end_idx] += gaussian_pulse[pulse_start:pulse_end]
    
    elif signal_type == 'B':
        # Signal with periodic components
        t = np.arange(N_L)
        signal = np.zeros(N_L)
        for _ in range(3):
            freq = np.random.uniform(1, 50)  # Increase frequency range
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.2, 1.0)
            signal += amplitude * np.sin(2*np.pi*freq*t/N_L + phase)
        signal += np.random.normal(0, 0.1, N_L)
    
    else:  # type C
        # Slowly varying trend signal
        t = np.arange(N_L)
        trend = 0.5 * np.sin(2*np.pi*t/N_L) + 0.3 * np.sin(4*np.pi*t/N_L)
        high_freq = 0.1 * np.sin(100*np.pi*t/N_L)  # Increase high-frequency component
        signal = trend + high_freq + np.random.normal(0, 0.1, N_L)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, 
        sr=sr, 
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Compute number of time frames
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    
    return mel_spec_db, n_frames

def evaluate_mel_preservation(original_mel, sampled_indices):
    """
    Evaluate Mel spectrogram preservation capability

    Args:
        original_mel (numpy.ndarray): Original Mel spectrogram
        sampled_indices (numpy.ndarray): Sampling indices

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Get sampled spectrogram (sampling along time dimension)
    sampled_mel = original_mel[:, sampled_indices]
    
    # Linear interpolation back to original time length
    n_mels, n_frames = original_mel.shape
    x_new = np.arange(n_frames)
    interpolated_mel = np.zeros_like(original_mel)
    
    # Interpolate each Mel band, handling boundary cases
    for i in range(n_mels):
        if len(sampled_indices) > 1:  # Ensure at least two points for interpolation
            indices = list(sampled_indices)
            values = list(sampled_mel[i])
            
            # Add boundary points only when needed
            if indices[0] > 0:
                indices.insert(0, 0)
                values.insert(0, original_mel[i, 0])
            
            if indices[-1] < n_frames - 1:
                indices.append(n_frames - 1)
                values.append(original_mel[i, -1])
            
            # Ensure no duplicate indices
            indices, unique_indices = np.unique(indices, return_index=True)
            values = np.array(values)[unique_indices]
            
            # Use cubic spline interpolation, fallback to linear if insufficient points
            if len(indices) >= 4:
                kind = 'cubic'
            else:
                kind = 'linear'
                
            f = interp1d(indices, values, kind=kind, bounds_error=False, fill_value="extrapolate")
            interpolated_mel[i] = f(x_new)
        else:
            # If only one sampling point, use constant interpolation
            interpolated_mel[i] = sampled_mel[i, 0]
    
    # Compute evaluation metrics
    mse = mean_squared_error(original_mel.flatten(), interpolated_mel.flatten())
    
    # Safely compute PSNR
    max_val = np.max(np.abs(original_mel))
    if mse > 0 and max_val > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')  # If MSE is 0, PSNR is infinite
    
    # Compute SSIM, ensure appropriate window size
    win_size = min(7, min(original_mel.shape) - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size >= 3:
        data_range = np.max(original_mel) - np.min(original_mel)
        ssim_value = ssim(original_mel, interpolated_mel, 
                         win_size=win_size, 
                         data_range=data_range,
                         channel_axis=None)
    else:
        ssim_value = 1.0 if np.array_equal(original_mel, interpolated_mel) else 0.0
    
    # Compute Frobenius norm distance
    frob_dist = np.linalg.norm(original_mel - interpolated_mel, ord='fro')
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Frobenius_Distance': frob_dist,
        'Interpolated_Mel': interpolated_mel
    }

def plot_mel_comparison(original_mel, sampling_results, signal_type=""):
    """
    Plot Mel spectrogram comparison
    """
    n_methods = len(sampling_results)
    plt.figure(figsize=(14, 6))
    
    # # Compute global max and min of all spectrograms
    # all_specs = [original_mel] + [result['Interpolated_Mel'] for result in sampling_results.values()]
    # vmin = min(spec.min() for spec in all_specs)
    # vmax = max(spec.max() for spec in all_specs)
    
    # Plot original Mel spectrogram
    plt.subplot(2, 2, 1)
    
    img = librosa.display.specshow(original_mel, y_axis='mel', x_axis='time')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Original Mel Spectrogram', fontproperties=get_times_bold_font(16))
    plt.xlabel('Time', fontproperties=get_times_font(14))
    plt.ylabel('Mel Frequency', fontproperties=get_times_font(14))
    
    # Plot reconstruction results for various methods
    for i, (method_name, result) in enumerate(sampling_results.items(), 2):
        plt.subplot(2, 2, i)
        img = librosa.display.specshow(result['Interpolated_Mel'], y_axis='mel', x_axis='time')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f'{method_name} Reconstructed (SSIM: {result["SSIM"]:.3f}, PSNR: {result["PSNR"]:.1f}dB)', fontproperties=get_times_bold_font(16))
        plt.xlabel('Time', fontproperties=get_times_font(14))
        plt.ylabel('Mel Frequency', fontproperties=get_times_font(14))
    
    plt.tight_layout()

def run_experiment(signal_type='A', N_L=22050, N_L_target=128, seeds=[7, 42, 1309]):
    """
    Run complete experiment
    """
    # Ensure output directory exists
    os.makedirs('code_for_essay/output_thesis', exist_ok=True)
    
    # Store results for all seeds
    all_results = {seed: {} for seed in seeds}
    
    # Set Mel spectrogram parameters
    n_fft = 1024
    hop_length = 64
    
    # Generate Mel spectrogram for first seed (for display)
    first_seed = seeds[0]
    display_mel, n_frames = generate_mel_spectrogram(
        signal_type, N_L, n_fft=n_fft, hop_length=hop_length, seed=first_seed
    )
    
    for seed in seeds:
        # Generate new Mel spectrogram for each seed
        original_mel, n_frames = generate_mel_spectrogram(
            signal_type, N_L, n_fft=n_fft, hop_length=hop_length, seed=seed
        )
        # print(original_mel.shape, N_L_target)
        
        # Compute target time frames, ensure not exceeding actual frames
        target_frames = min(N_L_target, n_frames - 1)
        
        # Apply different sampling methods
        sampling_methods = {
            'LDS': lambda: lds_sampling(n_frames, target_frames, seed=seed),
            'UDS': lambda: uniform_sampling(n_frames, target_frames),
            'SRS': lambda: random_sampling(n_frames, target_frames, seed=seed)
        }
        
        # Sample and evaluate each method
        for method_name, sampling_func in sampling_methods.items():
            indices = sampling_func()
            result = evaluate_mel_preservation(original_mel, indices)
            all_results[seed][method_name] = result
    
    # Compute average results (only average evaluation metrics)
    avg_results = {}
    metrics = ['MSE', 'PSNR', 'SSIM', 'Frobenius_Distance']
    
    for method in ['LDS', 'UDS', 'SRS']:
        # Use first seed's reconstruction result
        first_result = all_results[first_seed][method]
        avg_results[method] = {
            'MSE': 0,
            'PSNR': 0,
            'SSIM': 0,
            'Frobenius_Distance': 0,
            'Interpolated_Mel': first_result['Interpolated_Mel']  # Use first seed's reconstruction result
        }
        
        # Only average evaluation metrics
        for seed in seeds:
            for metric in metrics:
                avg_results[method][metric] += all_results[seed][method][metric]
        
        # Compute average of evaluation metrics
        n_seeds = len(seeds)
        for metric in metrics:
            avg_results[method][metric] /= n_seeds
    
    # Plot Mel spectrogram comparison (using first seed's reconstruction result)
    plot_mel_comparison(display_mel, avg_results, f"Type {signal_type}")
    
    # Save evaluation results
    preservation_data = []
    for method in ['LDS', 'UDS', 'SRS']:
        row = {'Method': method}
        for metric in metrics:
            if len(seeds) == 1:
                row[metric] = all_results[seeds[0]][method][metric]
            else:
                for seed in seeds:
                    row[f'{metric}_seed_{seed}'] = all_results[seed][method][metric]
                row[f'{metric}_avg'] = avg_results[method][metric]
        preservation_data.append(row)
    
    preservation_df = pd.DataFrame(preservation_data)
    preservation_df.to_csv(f'code_for_manu/output/figure7_type_{signal_type}.csv', index=False)

# Run experiment
if __name__ == "__main__":
    for signal_type in ['A', 'B', 'C']:
        run_experiment(signal_type=signal_type, 
                      N_L=22050,  # 1-second audio length
                      N_L_target=128,  # Target number of sampling points
                      seeds=[7,42,123,1309,5287,31415])
        plt.savefig(f'code_for_manu/output/figure7_type_{signal_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
