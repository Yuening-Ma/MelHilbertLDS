import os
import random
import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
import cv2

# Import Hilbert curve mapping functions
import sys
sys.path.append('gilbert/')
from gilbert2d import gilbert2d

'''
1. Dataset Configuration
'''

# Define supported dataset configurations
DATASET_CONFIGS = {
    'cough-speech-sneeze': {
        'dataset_path': 'datasets/cough-speech-sneeze/',
        'mel_dataset_path': 'datasets/cough-speech-sneeze_mel/',
        'categories': ['coughing', 'sneezing', 'speech'],
        'class_to_id': {
            'coughing': 0,
            'sneezing': 1,
            'speech': 2
        }
    },
    'CoughDataset': {
        'dataset_path': 'datasets/CoughDataset/',
        'mel_dataset_path': 'datasets/CoughDataset_mel/',
        'categories': ['covid', 'healthy', 'lower', 'obstructive', 'upper'],
        'class_to_id': {
            'covid': 0,
            'healthy': 1,
            'lower': 2,
            'obstructive': 3,
            'upper': 4
        }
    },
    'ESC50-human': {
        'dataset_path': 'datasets/ESC50-human/',
        'mel_dataset_path': 'datasets/ESC50-human_mel/',
        'categories': ['crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
                       'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping'],
        'class_to_id': {
            'crying_baby': 0,
            'sneezing': 1,
            'clapping': 2,
            'breathing': 3,
            'coughing': 4,
            'footsteps': 5,
            'laughing': 6,
            'brushing_teeth': 7,
            'snoring': 8,
            'drinking_sipping': 9
        }
    }
}


def get_dataset_config(dataset_name):
    """Get dataset configuration"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def create_train_val_index_files(dataset_name='cough-speech-sneeze', train_ratio=0.7):
    """
    Create training and validation set index files
    Split training and validation sets by category, ensuring each category's validation set
    accounts for (1-train_ratio) of that category's total count

    Args:
        dataset_name: Dataset name ('cough-speech-sneeze', 'CoughDataset' or 'ESC50-human')
        train_ratio: Proportion of training set to total data

    Returns:
        train_output_path, val_output_path: Training and validation index file paths
    """
    config = get_dataset_config(dataset_name)
    dataset_path = config['dataset_path']
    categories = config['categories']
    
    train_output_path = os.path.join(dataset_path, "train_index.txt")
    val_output_path = os.path.join(dataset_path, "val_index.txt")
    
    print(f"Creating training and validation index files for dataset '{dataset_name}'...")
    print(f"Training set ratio: {train_ratio*100:.0f}%, Validation set ratio: {(1-train_ratio)*100:.0f}%")
    
    train_files = []
    val_files = []
    
    # Collect and split files by category
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory does not exist {category_path}")
            continue
        
        # Collect all audio files for this category
        audio_files = [f for f in os.listdir(category_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
        category_files = [f"{category}/{file_name}" for file_name in audio_files]
        
        if not category_files:
            print(f"Warning: No audio files found in category '{category}'")
            continue
        
        # Shuffle files for this category
        random.shuffle(category_files)
        
        # Calculate the split point for this category
        category_total = len(category_files)
        category_train_size = int(category_total * train_ratio)
        
        # Split the data for this category
        category_train_files = category_files[:category_train_size]
        category_val_files = category_files[category_train_size:]
        
        train_files.extend(category_train_files)
        val_files.extend(category_val_files)
        
        print(f"  Category '{category}': total {category_total} files, "
              f"train {len(category_train_files)} ({len(category_train_files)/category_total*100:.0f}%), "
              f"val {len(category_val_files)} ({len(category_val_files)/category_total*100:.0f}%)")
    
    # Check if any files were collected
    if not train_files and not val_files:
        raise ValueError(f"No audio files found in dataset '{dataset_name}'")
    
    # Write training set index file
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for file_path in train_files:
            f.write(file_path + '\n')
    
    # Write validation set index file
    with open(val_output_path, 'w', encoding='utf-8') as f:
        for file_path in val_files:
            f.write(file_path + '\n')
    
    print(f"Index files created!")
    print(f"Training set: {len(train_files)} files ({train_ratio*100:.0f}%)")
    print(f"Validation set: {len(val_files)} files ({(1-train_ratio)*100:.0f}%)")
    print(f"Training set index file saved to: {os.path.abspath(train_output_path)}")
    print(f"Validation set index file saved to: {os.path.abspath(val_output_path)}")
    
    return train_output_path, val_output_path


'''
2. Feature Map Generation Functions
'''
def lds_sampling(original_length, target_length, start_index=0):
    """Low Discrepancy Sampling"""
    a = np.arange(1, original_length+1) * np.e % 1.0
    r = np.argsort(a)
    selected_indices = sorted(r[start_index:start_index+target_length])
    return selected_indices


def load_mel_spectrogram(audio_path, mel_dataset_path):
    """
    Load Mel spectrogram from precomputed .npy file

    Args:
        audio_path: Relative path of the original audio (e.g. coughing/xxx.wav)
        mel_dataset_path: Root directory path of the Mel dataset

    Returns:
        mel_spectrogram: Log-Mel spectrogram, shape (n_mels, time_frames)
    """
    # Convert audio path to .npy path
    base_name = os.path.splitext(audio_path)[0]
    npy_path = os.path.join(mel_dataset_path, base_name + '.npy')
    
    # Load precomputed Mel spectrogram
    mel_spectrogram = np.load(npy_path)
    
    return mel_spectrogram


def resize_mel_time_axis(mel_spectrogram, target_length):
    """
    Resize Mel spectrogram time axis using cv2.resize bilinear interpolation (interpolation approach)

    Args:
        mel_spectrogram: Mel spectrogram, shape (n_mels, time_frames)
        target_length: Target number of time frames

    Returns:
        resized_mel: Resized Mel spectrogram, shape (n_mels, target_length)
    """
    n_mels, time_frames = mel_spectrogram.shape
    
    if time_frames == target_length:
        return mel_spectrogram
    
    # cv2.resize expects input as (height, width), output as (new_height, new_width)
    # For mel spectrogram (n_mels, time_frames), we resize the time_frames dimension
    # Using INTER_LINEAR bilinear interpolation
    resized_mel = cv2.resize(
        mel_spectrogram.astype(np.float32),
        (target_length, n_mels),  # (width, height) = (time, freq)
        interpolation=cv2.INTER_LINEAR
    )
    
    return resized_mel


def sample_mel_lds(mel_spectrogram, target_length, start_index=0):
    """
    Sample Mel spectrogram time frames using Low Discrepancy Sampling (sampling approach)

    Args:
        mel_spectrogram: Mel spectrogram, shape (n_mels, time_frames)
        target_length: Target number of time frames
        start_index: LDS sampling start index, used for data augmentation

    Returns:
        sampled_mel: Sampled Mel spectrogram, shape (n_mels, target_length)
    """
    n_mels, time_frames = mel_spectrogram.shape
    
    if time_frames <= target_length:
        # Insufficient frames, apply padding
        if time_frames < target_length:
            padding = target_length - time_frames
            mel_spectrogram = np.pad(
                mel_spectrogram, 
                ((0, 0), (0, padding)), 
                mode='constant', 
                constant_values=mel_spectrogram.min()
            )
        return mel_spectrogram
    
    # Apply LDS sampling
    indices = lds_sampling(time_frames, target_length, start_index)
    sampled_mel = mel_spectrogram[:, indices]
    return sampled_mel


def normalize_mel(mel_spectrogram):
    """
    Normalize Mel spectrogram to [-1, 1] range

    Args:
        mel_spectrogram: Mel spectrogram

    Returns:
        normalized_mel: Normalized Mel spectrogram
    """
    min_val = np.min(mel_spectrogram) if np.min(mel_spectrogram) != 0 else -1e-5
    max_val = np.max(mel_spectrogram) if np.max(mel_spectrogram) != 0 else 1e-5
    normalized_mel = 2 * (mel_spectrogram - min_val) / (max_val - min_val) - 1
    return normalized_mel

def mel_to_hilbert(mel_spectrogram, hilbert_height=16, hilbert_width=16, hilbert_points=None):
    """
    Convert each row of the Mel spectrogram to a Hilbert curve-mapped image,
    forming a multi-channel feature map
    Fully vectorized implementation, truly loop-free

    Args:
        mel_spectrogram (numpy.ndarray): Mel spectrogram, shape (n_mels, time_frames)
        hilbert_height (int): Height of the Hilbert image
        hilbert_width (int): Width of the Hilbert image

    Returns:
        numpy.ndarray: Multi-channel feature map, shape (n_mels, hilbert_height, hilbert_width)
    """
    n_mels, time_frames = mel_spectrogram.shape
    
    # Check if the number of time frames equals the area of the Hilbert image
    if time_frames != hilbert_height * hilbert_width:
        raise ValueError(f"Mel spectrogram time frames ({time_frames}) must equal the Hilbert image area ({hilbert_height * hilbert_width})")
    
    # Generate points on the Hilbert curve
    if hilbert_points is None:
        hilbert_points = np.array(list(gilbert2d(hilbert_width, hilbert_height)))
    
    # Get Hilbert curve coordinates, limited to valid range
    y_coords = hilbert_points[:time_frames, 1]
    x_coords = hilbert_points[:time_frames, 0]
    
    # Create linear indices
    linear_indices = y_coords * hilbert_width + x_coords
    
    # Create index array for reordering
    # This is a permutation index from mel spectrogram to Hilbert curve mapping
    # For each frequency band, we need to reorder its values according to linear_indices
    permutation = np.zeros(hilbert_height * hilbert_width, dtype=np.int64)
    permutation[linear_indices] = np.arange(time_frames)
    
    # Rearrange each mel frequency band's values into a linear array, then reshape to 2D image
    # First reshape mel_spectrogram to fully flattened form
    flat_mel = mel_spectrogram.reshape(n_mels, -1)
    
    # Reorder values for each frequency band using the same permutation index
    # Use advanced indexing to complete all frequency band reordering at once
    # np.arange(n_mels)[:, np.newaxis] creates an index array of shape (n_mels, 1)
    # This works with permutation[np.newaxis, :] (shape 1, time_frames) to perform broadcasting
    reordered = flat_mel[np.arange(n_mels)[:, np.newaxis], permutation[np.newaxis, :]]
    
    # Reshape to final multi-channel feature map
    feature_map = reordered.reshape(n_mels, hilbert_height, hilbert_width)
    
    return feature_map

def mel_to_hilbert_time(mel_spectrogram, hilbert_height=16, hilbert_width=16, hilbert_points=None):
    """
    Convert each column (time frame) of the Mel spectrogram to a Hilbert curve-mapped image,
    forming a multi-channel feature map

    Args:
        mel_spectrogram (numpy.ndarray): Mel spectrogram, shape (n_mels, time_frames)
        hilbert_height (int): Height of the Hilbert image
        hilbert_width (int): Width of the Hilbert image

    Returns:
        numpy.ndarray: Multi-channel feature map, shape (time_frames, hilbert_height, hilbert_width)
    """
    n_mels, time_frames = mel_spectrogram.shape
    
    # Check if the number of frequency channels equals the area of the Hilbert image
    if n_mels != hilbert_height * hilbert_width:
        raise ValueError(f"Mel spectrogram frequency channels ({n_mels}) must equal the Hilbert image area ({hilbert_height * hilbert_width})")
    
    # Generate points on the Hilbert curve
    if hilbert_points is None:
        hilbert_points = np.array(list(gilbert2d(hilbert_width, hilbert_height)))
    
    # Get Hilbert curve coordinates, limited to valid range
    y_coords = hilbert_points[:n_mels, 1]
    x_coords = hilbert_points[:n_mels, 0]
    
    # Create linear indices
    linear_indices = y_coords * hilbert_width + x_coords
    
    # Create index array for reordering
    # This is a permutation index from mel spectrogram to Hilbert curve mapping
    # For each time frame, we need to reorder its values according to linear_indices
    permutation = np.zeros(hilbert_height * hilbert_width, dtype=np.int64)
    permutation[linear_indices] = np.arange(n_mels)
    
    # Rearrange each time frame's values into a linear array, then reshape to 2D image
    # First reshape mel_spectrogram to fully flattened form
    flat_mel = mel_spectrogram.reshape(-1, time_frames)
    
    # Reorder values for each time frame using the same permutation index
    # Use advanced indexing to complete all time frame reordering at once
    # np.arange(time_frames)[np.newaxis, :] creates an index array of shape (1, time_frames)
    # This works with permutation[:, np.newaxis] (shape n_mels, 1) to perform broadcasting
    reordered = flat_mel[permutation[:, np.newaxis], np.arange(time_frames)[np.newaxis, :]]
    
    # Reshape to final multi-channel feature map
    feature_map = reordered.reshape(hilbert_height, hilbert_width, time_frames)
    
    # Adjust dimension order so that time frames become the channel dimension
    feature_map = np.transpose(feature_map, (2, 0, 1))
    
    return feature_map

def audio_to_signal(audio, target_length=None):
    """
    Adjust a 1D audio sequence to the specified length

    Args:
        audio (numpy.ndarray): Audio sequence
        sr (int): Sample rate, default 16000
        target_length (int): Optional, specify target length; if None, use len(audio)

    Returns:
        numpy.ndarray: Adjusted audio sequence, shape (target_length,)
    """
    # Calculate target length
    if target_length is None:
        target_length = len(audio)
    
    # Adjust audio length
    original_length = len(audio)
    if original_length > target_length:
        # If audio length exceeds target, truncate from the beginning
        sampled_audio = audio[:target_length]
    elif original_length < target_length:
        # If audio length is less than target, pad with zeros at the end
        padding = target_length - original_length
        sampled_audio = np.pad(audio, (0, padding), mode='constant')
    else:
        # Length matches exactly
        sampled_audio = audio   
    
    return sampled_audio

def audio_to_signal_lds(audio, target_length=None, start_index=0):
    """
    Adjust a 1D audio sequence to the specified length using LDS

    Args:
        audio (numpy.ndarray): Audio sequence
        sr (int): Sample rate, default 16000
        target_length (int): Optional, specify target length; if None, use len(audio)

    Returns:
        numpy.ndarray: Adjusted audio sequence, shape (target_length,)
    """
    # Calculate target length
    if target_length is None:
        target_length = len(audio)
    
    # Adjust audio length
    original_length = len(audio)
    if original_length > target_length:
        # If audio length exceeds target, use LDS sampling
        indices = lds_sampling(original_length, target_length, start_index)
        sampled_audio = audio[indices]
    elif original_length < target_length:
        # If audio length is less than target, pad with zeros at the end
        padding = target_length - original_length
        sampled_audio = np.pad(audio, (0, padding), mode='constant')
    else:
        # Length matches exactly
        sampled_audio = audio   
    
    return sampled_audio

def audio_to_hilbert(audio_sequence, hilbert_height=64, hilbert_width=64, target_length=None, hilbert_points=None):
    """
    Map a 1D audio sequence to a 2D image via generalized Hilbert curve

    Args:
        audio_sequence (numpy.ndarray): Audio sequence
        height (int): Output image height, default 64
        width (int): Output image width, default 64
        target_length (int): Optional, specify target length; if None, use height*width
        hilbert_points (numpy.ndarray): Optional, precomputed Hilbert curve points

    Returns:
        numpy.ndarray: 2D array representing the converted image, shape (height, width)
    """
    # Calculate target length
    if target_length is None:
        target_length = hilbert_height * hilbert_width
    
    # Adjust audio length
    original_length = len(audio_sequence)
    if original_length > target_length:
        # If audio length exceeds target, truncate from the beginning
        sampled_audio = audio_sequence[:target_length]
    elif original_length < target_length:
        # If audio length is less than target, pad with zeros at the end
        padding = target_length - original_length
        sampled_audio = np.pad(audio_sequence, (0, padding), 'constant')
    else:
        # Length matches exactly
        sampled_audio = audio_sequence
    
    # Ensure sampled_audio length equals target_length
    assert len(sampled_audio) == target_length, f"Sampled audio length should be {target_length}, but got {len(sampled_audio)}"
    
    # Generate points on the Hilbert curve
    if hilbert_points is None:
        hilbert_points = np.array(list(gilbert2d(hilbert_width, hilbert_height)))
    
    # Create blank image
    image = np.zeros((hilbert_height, hilbert_width), dtype=np.float32)
    
    # Vectorized operation: fill image using advanced indexing directly
    y_coords = hilbert_points[:len(sampled_audio), 1]
    x_coords = hilbert_points[:len(sampled_audio), 0]
    image[y_coords, x_coords] = sampled_audio
    
    # Normalize to [-1, 1] range
    min_val = np.min(image) if np.min(image) != 0 else -1e-5
    max_val = np.max(image) if np.max(image) != 0 else 1e-5
    image = 2 * (image - min_val) / (max_val - min_val) - 1
    
    return image

def audio_to_hilbert_lds(audio_sequence, hilbert_height=64, hilbert_width=64, target_length=None, hilbert_points=None, start_index=0):
    """
    Map a 1D audio sequence to a 2D image via generalized Hilbert curve, using LDS sampling for downsampling

    Args:
        audio_sequence (numpy.ndarray): Audio sequence
        height (int): Output image height, default 64
        width (int): Output image width, default 64
        target_length (int): Optional, specify target length; if None, use height*width
        hilbert_points (numpy.ndarray): Optional, precomputed Hilbert curve points
        start_index (int): LDS sampling start index, used for data augmentation

    Returns:
        numpy.ndarray: 2D array representing the converted image, shape (height, width)
    """
    # Calculate target length
    if target_length is None:
        target_length = hilbert_height * hilbert_width
    
    # Adjust audio length
    original_length = len(audio_sequence)
    if original_length > target_length:
        # If audio length exceeds target, use LDS sampling for extraction
        indices = lds_sampling(original_length, target_length, start_index)
        sampled_audio = audio_sequence[indices]
    elif original_length < target_length:
        # If audio length is less than target, pad with zeros at the end
        padding = target_length - original_length
        sampled_audio = np.pad(audio_sequence, (0, padding), mode='constant')
    else:
        # Length matches exactly
        sampled_audio = audio_sequence
    
    # Ensure sampled_audio length equals target_length
    assert len(sampled_audio) == target_length, f"Sampled audio length should be {target_length}, but got {len(sampled_audio)}"
    
    # Generate points on the Hilbert curve
    if hilbert_points is None:
        hilbert_points = np.array(list(gilbert2d(hilbert_width, hilbert_height)))
    
    # Create blank image
    image = np.zeros((hilbert_height, hilbert_width), dtype=np.float32)
    
    # Vectorized operation: fill image using advanced indexing directly
    y_coords = hilbert_points[:len(sampled_audio), 1]
    x_coords = hilbert_points[:len(sampled_audio), 0]
    image[y_coords, x_coords] = sampled_audio
    
    # Normalize to [-1, 1] range
    min_val = np.min(image) if np.min(image) != 0 else -1e-5
    max_val = np.max(image) if np.max(image) != 0 else 1e-5
    image = 2 * (image - min_val) / (max_val - min_val) - 1
    
    return image


'''
3. Create Dataset
'''
class CSSDataset(data.Dataset):
    def __init__(
        self, mode='train', 
        transform=None, 
        data_augmentation=True, 
        type='Mel', # Optional values: 'Mel', 'MelHilbert', 'SignalHilbert', 'SignalHilbertLDS',
                    # 'MelLDS', 'MelLDS', 'MelHilbertLDS', 'MelHilbertLDS',
                    # 'MelHilbertTime', 'MelHilbertTimeLDS', 'MelHilbertTimeLDS'
        dataset_name='cough-speech-sneeze',  # Optional values: 'cough-speech-sneeze', 'CoughDataset', 'ESC50-human'
        n_mels=128, 
        mel_time_frames=256, 
        fixed_length=128,
        hilbert_height=16, 
        hilbert_width=16, 
        height=128, 
        width=128,
        hop_length=64, 
        n_fft=1024, 
        seed=42,
        signal_length=1024,
        **kwargs
    ):
        """
        Initialize the general cough sound dataset

        Args:
            mode (str): 'train' or 'val', specify whether to use training or validation set
            transform (callable, optional): Optional data transformation
            data_augmentation (bool): Whether to apply data augmentation, default False
            type (str): Dataset type, determines which feature extraction method to use
            dataset_name (str): Dataset name, 'cough-speech-sneeze', 'CoughDataset' or 'ESC50-human'
            n_mels (int): Number of Mel filter banks
            mel_time_frames (int): Number of time frames in the Mel spectrogram
            fixed_length (int): Fixed number of time frames, used for padding or truncation
            hilbert_height (int): Height of the Hilbert image
            hilbert_width (int): Width of the Hilbert image
            height (int): Output image height
            width (int): Output image width
            hop_length (int): Hop length
            n_fft (int): FFT length
            seed (int): Random seed
            **kwargs: Other parameters
        """
        self.mode = mode
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.type = type
        self.dataset_name = dataset_name
        
        # Get dataset configuration
        config = get_dataset_config(dataset_name)
        self.dataset_path = config['dataset_path']
        self.mel_dataset_path = config['mel_dataset_path']
        self.categories = config['categories']
        self.class_to_id = config['class_to_id']
        
        self.n_mels = n_mels
        self.mel_time_frames = mel_time_frames
        self.fixed_length = fixed_length
        self.hilbert_height = hilbert_height
        self.hilbert_width = hilbert_width
        self.height = height
        self.width = width
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.signal_length = signal_length
        self.rng = torch.Generator().manual_seed(seed)

        if 'Hilbert' in type:
            self.hilbert_points = np.array(list(gilbert2d(hilbert_width, hilbert_height)))
        
        # Validate parameters, set Hilbert points
        if type in ['MelHilbertLDS', 'MelHilbertLDS']:

            if self.mel_time_frames != self.hilbert_height * self.hilbert_width:
                raise ValueError(f"Mel spectrogram time frames ({self.mel_time_frames}) must equal the Hilbert image area ({self.hilbert_height * self.hilbert_width})")
        
        if type in ['MelHilbertTime', 'MelHilbertTimeLDS', 'MelHilbertTimeLDS']:

            if self.n_mels != self.hilbert_height * self.hilbert_width:
                raise ValueError(f"Mel spectrogram frequency channels ({self.n_mels}) must equal the Hilbert image area ({self.hilbert_height * self.hilbert_width})")
            
        # Select index file based on mode
        if mode == 'train':
            index_file = os.path.join(self.dataset_path, "train_index.txt")
        elif mode == 'val':
            index_file = os.path.join(self.dataset_path, "val_index.txt")
        else:
            raise ValueError("Mode must be 'train' or 'val'")
        
        # Read file list
        self.file_list = []
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.file_list.append(line.strip())
    
    def __len__(self):
        """Return dataset size"""
        return len(self.file_list)
    
    def apply_time_shift(self, audio, sr):
        """Random time shift"""
        if len(audio) > sr:  # Ensure audio is longer than 1 second
            max_shift = min(len(audio) - sr, sr)  # Shift at most 1 second but not exceeding remaining length
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:
                audio = np.pad(audio, (0, shift), 'constant')[shift:]
            elif shift < 0:
                audio = np.pad(audio, (-shift, 0), 'constant')[:shift]
        return audio
    
    def apply_pitch_shift(self, audio, sr):
        """Random pitch shift"""
        n_steps = np.random.uniform(-4, 4)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def apply_time_stretch(self, audio):
        """Random time stretch"""
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def apply_noise(self, audio):
        """Add random noise"""
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level, size=len(audio))
        return audio + noise
    
    def apply_volume_change(self, audio):
        """Random volume change"""
        volume_factor = np.random.uniform(0.75, 1.25)
        return audio * volume_factor
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Get audio file path
        audio_path = self.file_list[idx]
        
        # Extract category name from file path (format: category/xxx.wav)
        category = audio_path.split('/')[0]  # Get category name
        # Use class_to_id dictionary to map category name to corresponding ID
        label = self.class_to_id[category]
        
        # Determine if it's a Mel-related type (needs to be loaded from precomputed .npy)
        mel_types = ['Mel', 'MelLDS', 
                     'MelHilbert', 'MelHilbertLDS',
                     'MelHilbertTime', 'MelHilbertTimeLDS']
        
        # Select feature extraction method based on type
        # ========== Interpolation approach: use cv2.resize bilinear interpolation ==========
        if self.type == 'Mel':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            # Resize time axis using bilinear interpolation
            mel_spectrogram = resize_mel_time_axis(mel_spectrogram, self.fixed_length)
            # Normalize
            feature = normalize_mel(mel_spectrogram)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            
        elif self.type == 'MelHilbert':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            # Resize time axis using bilinear interpolation
            mel_spectrogram = resize_mel_time_axis(mel_spectrogram, self.mel_time_frames)
            # Normalize
            mel_spectrogram = normalize_mel(mel_spectrogram)
            # Hilbert curve mapping
            feature = mel_to_hilbert(
                mel_spectrogram,
                hilbert_height=self.hilbert_height,
                hilbert_width=self.hilbert_width,
                hilbert_points=self.hilbert_points
            )
            feature_tensor = torch.FloatTensor(feature)
            
        elif self.type == 'MelHilbertTime':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            # Resize time axis using bilinear interpolation
            mel_spectrogram = resize_mel_time_axis(mel_spectrogram, self.mel_time_frames)
            # Normalize
            mel_spectrogram = normalize_mel(mel_spectrogram)
            # Hilbert time-axis mapping
            feature = mel_to_hilbert_time(
                mel_spectrogram,
                hilbert_height=self.hilbert_height,
                hilbert_width=self.hilbert_width,
                hilbert_points=self.hilbert_points
            )
            feature_tensor = torch.FloatTensor(feature)
            
        # ========== Sampling approach: use LDS frame extraction ==========
        elif self.type == 'MelLDS':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            n_frames = mel_spectrogram.shape[1]
            
            if self.mode == 'train' and n_frames > self.fixed_length:
                # Training mode: LDS frame extraction, random start index for data augmentation
                start_idx = torch.randint(0, n_frames - self.fixed_length + 1, (1,), generator=self.rng).item()
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.fixed_length, start_idx)
            else:
                # Validation mode: LDS frame extraction, fixed start index
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.fixed_length, start_index=0)
            
            # Normalize
            feature = normalize_mel(mel_spectrogram)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        elif self.type == 'MelHilbertLDS':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            n_frames = mel_spectrogram.shape[1]
            
            if self.mode == 'train' and n_frames > self.mel_time_frames:
                # Training mode: LDS frame extraction, random start index for data augmentation
                start_idx = torch.randint(0, n_frames - self.mel_time_frames + 1, (1,), generator=self.rng).item()
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.mel_time_frames, start_idx)
            else:
                # Validation mode: LDS frame extraction, fixed start index
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.mel_time_frames, start_index=0)
            
            # Normalize
            mel_spectrogram = normalize_mel(mel_spectrogram)
            # Hilbert curve mapping
            feature = mel_to_hilbert(
                mel_spectrogram, 
                hilbert_height=self.hilbert_height, 
                hilbert_width=self.hilbert_width,
                hilbert_points=self.hilbert_points
            )
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            
        elif self.type == 'MelHilbertTimeLDS':
            # Load Mel spectrogram from precomputed .npy
            mel_spectrogram = load_mel_spectrogram(audio_path, self.mel_dataset_path)
            n_frames = mel_spectrogram.shape[1]
            
            if self.mode == 'train' and n_frames > self.mel_time_frames:
                # Training mode: LDS frame extraction, random start index for data augmentation
                start_idx = torch.randint(0, n_frames - self.mel_time_frames + 1, (1,), generator=self.rng).item()
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.mel_time_frames, start_idx)
            else:
                # Validation mode: LDS frame extraction, fixed start index
                mel_spectrogram = sample_mel_lds(mel_spectrogram, self.mel_time_frames, start_index=0)
            
            # Normalize
            mel_spectrogram = normalize_mel(mel_spectrogram)
            # Hilbert time-axis mapping
            feature = mel_to_hilbert_time(
                mel_spectrogram, 
                hilbert_height=self.hilbert_height, 
                hilbert_width=self.hilbert_width,
                hilbert_points=self.hilbert_points
            )
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            
        # ========== Signal-related types: load from audio files ==========
        elif self.type == 'SignalHilbert':
            # Load audio file
            audio, sr = librosa.load(os.path.join(self.dataset_path, audio_path), sr=None)
            audio = np.clip(audio, -1, 1)
            
            target_length = self.height * self.width
            # Recalculate sr so that the loaded audio length equals target_length
            target_sr = int(sr * target_length / len(audio))
            audio, _ = librosa.load(os.path.join(self.dataset_path, audio_path), sr=target_sr)
            feature = audio_to_hilbert(
                audio,
                hilbert_height=self.hilbert_height,
                hilbert_width=self.hilbert_width,
                target_length=target_length,
                hilbert_points=self.hilbert_points
            )
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            
        elif self.type == 'SignalHilbertLDS':
            # Load audio file
            audio, sr = librosa.load(os.path.join(self.dataset_path, audio_path), sr=None)
            audio = np.clip(audio, -1, 1)
            
            target_length = self.height * self.width
            if self.mode == 'train' and len(audio) > target_length:
                max_start = len(audio) - target_length + 1
                start_idx = torch.randint(0, max_start, (1,), generator=self.rng).item()
            else:
                start_idx = 0
            feature = audio_to_hilbert_lds(
                audio_sequence=audio,
                hilbert_height=self.hilbert_height,
                hilbert_width=self.hilbert_width,
                target_length=target_length,
                hilbert_points=self.hilbert_points,
                start_index=start_idx
            )
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            
        elif self.type == 'Signal':
            # Load audio file
            audio, sr = librosa.load(os.path.join(self.dataset_path, audio_path), sr=None)
            audio = np.clip(audio, -1, 1)
            
            target_length = self.signal_length
            # Recalculate sr so that the loaded audio length equals target_length
            target_sr = int(sr * target_length / len(audio))
            audio, _ = librosa.load(os.path.join(self.dataset_path, audio_path), sr=target_sr)
            feature = audio_to_signal(audio, target_length=target_length)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            
        elif self.type == 'SignalLDS':
            # Load audio file
            audio, sr = librosa.load(os.path.join(self.dataset_path, audio_path), sr=None)
            audio = np.clip(audio, -1, 1)
            
            target_length = self.signal_length
            if self.mode == 'train' and len(audio) > target_length:
                max_start = len(audio) - target_length + 1
                start_idx = torch.randint(0, max_start, (1,), generator=self.rng).item()
            else:
                start_idx = 0
            feature = audio_to_signal_lds(audio, target_length=target_length, start_index=start_idx)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        else:
            raise ValueError(f"Unsupported dataset type: {self.type}")
        
        # Apply transformation (if any)
        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        
        return feature_tensor, torch.tensor(label, dtype=torch.long)


'''
4. Create Data Loader
'''
def get_data_loader(mode='train', batch_size=32, num_workers=4, data_augmentation=True,
                   type='Mel',  # Optional values: 'Mel', 'MelHilbert', 'SignalHilbert', 'SignalHilbertLDS',
                              # 'MelLDS', 'MelLDS', 'MelHilbertLDS', 'MelHilbertLDS',
                              # 'MelHilbertTime', 'MelHilbertTimeLDS', 'MelHilbertTimeLDS'
                   dataset_name='cough-speech-sneeze',  # Optional values: 'cough-speech-sneeze', 'CoughDataset', 'ESC50-human'
                   n_mels=128, mel_time_frames=256, fixed_length=128,
                   hilbert_height=16, hilbert_width=16, 
                   height=128, width=128,
                   hop_length=64, n_fft=1024, seed=42,
                   num_samples=12800,**kwargs):
    """
    Create a general data loader using the CSSDataset class

    Args:
        mode (str): 'train' or 'val'
        batch_size (int): Batch size
        num_workers (int): Number of worker threads for data loading
        data_augmentation (bool): Whether to apply data augmentation, default False
        type (str): Dataset type, determines which feature extraction method to use
        dataset_name (str): Dataset name, 'cough-speech-sneeze', 'CoughDataset' or 'ESC50-human'
        n_mels (int): Number of Mel filter banks
        mel_time_frames (int): Number of time frames in the Mel spectrogram
        fixed_length (int): Fixed number of time frames, used for padding or truncation
        hilbert_height (int): Height of the Hilbert image
        hilbert_width (int): Width of the Hilbert image
        height (int): Output image height
        width (int): Output image width
        hop_length (int): Hop length
        n_fft (int): FFT length
        seed (int): Random seed
        num_samples (int): Number of samples per epoch (only used in training mode)

    Returns:
        data_loader: DataLoader object
    """
    
    # Create dataset instance
    dataset = CSSDataset(
        mode=mode,
        data_augmentation=data_augmentation,
        type=type,
        dataset_name=dataset_name,
        n_mels=n_mels,
        mel_time_frames=mel_time_frames,
        fixed_length=fixed_length,
        hilbert_height=hilbert_height,
        hilbert_width=hilbert_width,
        height=height,
        width=width,
        hop_length=hop_length,
        n_fft=n_fft,
        seed=seed,
        **kwargs
    )
    
    if mode == 'train':
        # Calculate sample count for each class
        class_counts = {}
        for file_path in dataset.file_list:
            category = file_path.split('/')[0]
            if category not in class_counts:
                class_counts[category] = 0
            class_counts[category] += 1
        
        # Calculate weight for each class (inverse of sample count)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        total_weight = sum(class_weights.values())
        # Normalize weights
        class_weights = {cls: weight/total_weight*len(class_counts) for cls, weight in class_weights.items()}
        
        # Assign weights to each sample
        weights = []
        for file_path in dataset.file_list:
            category = file_path.split('/')[0]
            weights.append(class_weights[category])
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            # sampler=sampler,  # Use weighted sampler
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
    else:
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False, 
            shuffle=False,
        )
    
    return data_loader


