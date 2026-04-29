import os
import sys
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

# Set parameters
N_FFT = 1024
HOP_LENGTH = 64
N_MELS = 128
WIN_LENGTH = 1024

# Dataset paths
# DATASET_PATH = "datasets/cough-speech-sneeze/"
# OUTPUT_PATH = "datasets/cough-speech-sneeze_mel/"
# DATASET_PATH = "datasets/CoughDataset/"
# OUTPUT_PATH = "datasets/CoughDataset_mel/"
DATASET_PATH = "datasets/ESC50-human"
OUTPUT_PATH = "datasets/ESC50-human_mel"

# Supported audio formats
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.wma')


def extract_mel_spectrogram(audio_path):
    """
    Extract Mel spectrogram from audio file

    Args:
        audio_path: Audio file path

    Returns:
        mel_spectrogram_db: Log-Mel spectrogram, shape (n_mels, time_frames)
        sr: Sample rate
    """
    # Load audio using original sample rate
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        center=False  # Disable center padding to ensure accurate frame count calculation
    )
    
    # Convert to dB scale (Log-Mel)
    # ref=np.max uses maximum value as reference
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db, sr


def process_audio_file(audio_path, output_path):
    """
    Process a single audio file and save Mel spectrogram

    Args:
        audio_path: Audio file path
        output_path: Output .npy file path

    Returns:
        success: Whether processing succeeded
    """
    try:
        # Extract Mel spectrogram
        mel_spectrogram, sr = extract_mel_spectrogram(audio_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as .npy file
        np.save(output_path, mel_spectrogram)
        
        return True
        
    except Exception as e:
        print(f"\n  Error: Processing {audio_path} failed: {e}")
        return False


def get_relative_path(file_path, base_path):
    """
    Get relative path with respect to base_path

    Args:
        file_path: Full file path
        base_path: Base path

    Returns:
        relative_path: Relative path
    """
    return os.path.relpath(file_path, base_path)


def get_output_npy_path(audio_path, dataset_path, output_path):
    """
    Generate corresponding output .npy path based on audio path

    Args:
        audio_path: Audio file path
        dataset_path: Dataset root directory
        output_path: Output root directory

    Returns:
        npy_path: Output .npy file path
    """
    # Get relative path
    rel_path = get_relative_path(audio_path, dataset_path)
    
    # Replace extension with .npy
    base_name = os.path.splitext(rel_path)[0]
    npy_name = base_name + '.npy'
    
    # Combine output path
    return os.path.join(output_path, npy_name)


def collect_audio_files(dataset_path):
    """
    Collect all audio files in the dataset

    Args:
        dataset_path: Dataset root directory

    Returns:
        audio_files: List of audio file paths
    """
    audio_files = []
    
    # Traverse dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_path = os.path.join(root, file)
                audio_files.append(audio_path)
    
    return audio_files


def get_category_from_path(file_path, dataset_path):
    """
    Get category name from file path

    Args:
        file_path: File path
        dataset_path: Dataset root directory

    Returns:
        category: Category name (coughing/sneezing/speech)
    """
    rel_path = get_relative_path(file_path, dataset_path)
    category = rel_path.split(os.sep)[0]
    return category


def main():
    """Main function"""
    print("Mel Spectrogram Precomputation Script")
    print("=" * 60)
    print()
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print()
    print(f"Parameter settings:")
    print(f"  n_fft: {N_FFT}")
    print(f"  hop_length: {HOP_LENGTH}")
    print(f"  n_mels: {N_MELS}")
    print(f"  win_length: {WIN_LENGTH}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)
    
    # Collect all audio files
    print("Scanning audio files...")
    audio_files = collect_audio_files(DATASET_PATH)
    
    if len(audio_files) == 0:
        print("Error: No audio files found")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files")
    print()
    
    # Count files per category
    category_counts = {}
    for audio_path in audio_files:
        category = get_category_from_path(audio_path, DATASET_PATH)
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("Category distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} files")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Process all audio files
    print("Starting to process audio files...")
    success_count = 0
    fail_count = 0
    
    # Use tqdm to show progress
    for audio_path in tqdm(audio_files, desc="Processing progress"):
        # Generate output path
        npy_path = get_output_npy_path(audio_path, DATASET_PATH, OUTPUT_PATH)
    
        # Check if already exists
        if os.path.exists(npy_path):
            # Skip already processed files
            success_count += 1
            continue
    
        # Process audio file
        if process_audio_file(audio_path, npy_path):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Success: {success_count} files")
    print(f"Failed: {fail_count} files")
    print(f"Output directory: {os.path.abspath(OUTPUT_PATH)}")
    print()
    
    # Show some statistics
    if success_count > 0:
        print("Sample output files:")
        sample_files = []
        for root, dirs, files in os.walk(OUTPUT_PATH):
            for file in files:
                if file.endswith('.npy'):
                    sample_files.append(os.path.join(root, file))
                    if len(sample_files) >= 3:
                        break
            if len(sample_files) >= 3:
                break
    
        for sample_file in sample_files[:3]:
            data = np.load(sample_file)
            rel_path = get_relative_path(sample_file, OUTPUT_PATH)
            print(f"  {rel_path}: shape={data.shape}, dtype={data.dtype}")
    
    print()


if __name__ == '__main__':
    main()
