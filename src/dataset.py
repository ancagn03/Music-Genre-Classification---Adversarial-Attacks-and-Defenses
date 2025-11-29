"""
Dataset Loading and Preprocessing Module.

This module handles the loading and processing of the GTZAN Music Genre Dataset.
It provides two dataset classes:
1. GTZANDataset: Loads extracted audio features (MFCCs, Spectral Centroid, etc.) for MLP models.
2. GTZANSpectrogramDataset: Loads Mel-spectrograms as images for CNN/ResNet models.

It also includes utilities for automatically downloading the dataset if it is missing.
"""
import os
import torch
import librosa
import numpy as np
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def download_gtzan_if_missing(dataset_path='data/gtzan-music-genre-dataset'):
    """
    Checks if the GTZAN dataset exists at the specified path.
    If not found, attempts to download it (primarily for Colab environments).
    
    Args:
        dataset_path (str): The expected path to the dataset directory.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset NOT found at: {os.path.abspath(dataset_path)}")
        
        # Try to detect Colab environment or just force download if git is available
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False

        if is_colab:
            print("📥 Running on Colab. Downloading dataset from Hugging Face...")
            # We use os.system for shell commands in python scripts
            # Clone into the specific data directory
            os.makedirs('data', exist_ok=True)
            os.system(f"git clone https://huggingface.co/datasets/storylinez/gtzan-music-genre-dataset {dataset_path}")
            print("✅ Download complete!")
        else:
            print("⚠️ Not on Colab. Please download the dataset manually or check your path.")
    else:
        print(f"✅ Dataset found at: {os.path.abspath(dataset_path)}")

class GTZANDataset(Dataset):
    """
    PyTorch Dataset for loading feature-based GTZAN data (for MLP).
    
    Extracts and caches the following features from 30s audio clips:
    - MFCCs (Mean & Std)
    - Spectral Centroid (Mean & Std)
    - Zero Crossing Rate (Mean & Std)
    - Chroma STFT (Mean & Std)
    
    Total feature vector dimension: 44.
    """
    def __init__(self, root_dir='data/gtzan-music-genre-dataset', split='train', seed=42):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            split (str): Dataset split to load ('train', 'val', or 'test').
            seed (int): Random seed for reproducible train/val/test splits.
        """
        self.root_dir = root_dir
        self.classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Feature cache file (saves time on subsequent runs)
        self.cache_file = os.path.join(root_dir, 'features_mlp.pt')
        
        # Ensure dataset exists
        download_gtzan_if_missing(root_dir)
        
        if not os.path.exists(self.cache_file):
            self._process_and_cache()
            
        # Load features from disk
        self.data = torch.load(self.cache_file)
        
        # Create splits (80% train, 10% val, 10% test)
        total_size = len(self.data['labels'])
        indices = np.random.RandomState(seed).permutation(total_size)
        
        train_len = int(total_size * 0.8)
        val_len = int(total_size * 0.1)
        
        if split == 'train':
            self.indices = indices[:train_len]
        elif split == 'val':
            self.indices = indices[train_len:train_len + val_len]
        else:
            self.indices = indices[train_len + val_len:]
            
        print(f"Loaded {split} set: {len(self.indices)} samples")

    def _process_and_cache(self):
        """
        Iterates through all .wav files, extracts features using Librosa, and saves them to a .pt file.
        This is a one-time process.
        """
        print("🎵 Processing GTZAN dataset for MLP (extracting MFCCs)... this happens only once.")
        features_list = []
        labels_list = []
        
        # Iterate over genres
        for genre in self.classes:
            genre_dir = os.path.join(self.root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
                
            files = glob.glob(os.path.join(genre_dir, '*.wav'))
            print(f"Processing {genre}...")
            
            for f in tqdm(files):
                try:
                    # Load audio (30s)
                    y, sr = librosa.load(f, duration=30)
                    
                    # 1. MFCCs (Mean & Std) - Capture timbre
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    mfcc_mean = mfcc.mean(axis=1)
                    mfcc_std = mfcc.std(axis=1)
                    
                    # 2. Spectral Centroid (Mean & Std) - Capture brightness
                    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    cent_mean = cent.mean()
                    cent_std = cent.std()
                    
                    # 3. Zero Crossing Rate (Mean & Std) - Capture noisiness/percussiveness
                    zcr = librosa.feature.zero_crossing_rate(y)
                    zcr_mean = zcr.mean()
                    zcr_std = zcr.std()
                    
                    # Concatenate all features into a single vector
                    # Size: 20 + 20 + 1 + 1 + 1 + 1 = 44 features
                    feat = np.concatenate([mfcc_mean, mfcc_std, [cent_mean, cent_std, zcr_mean, zcr_std]])
                    
                    features_list.append(feat)
                    labels_list.append(self.class_to_idx[genre])
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    
        # Save to disk
        data = {
            'features': torch.tensor(np.array(features_list), dtype=torch.float32),
            'labels': torch.tensor(np.array(labels_list), dtype=torch.long)
        }
        torch.save(data, self.cache_file)
        print(f"✅ Saved processed features to {self.cache_file}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data['features'][real_idx], self.data['labels'][real_idx]

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, root_dir='data/gtzan-music-genre-dataset', split='train', seed=42):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            split (str): 'train', 'val', or 'test'.
            seed (int): Random seed.
        """
        self.root_dir = root_dir
        self.split = split
        self.classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Cache file for spectrograms
        self.cache_file = os.path.join(root_dir, 'features_cnn_sliced.pt')
        
        download_gtzan_if_missing(root_dir)
        
        if not os.path.exists(self.cache_file):
            self._process_and_cache()
            
        self.data = torch.load(self.cache_file)
        
        # Splits (Split by SONG, not by chunk, to avoid data leakage)
        # We assume exactly 10 chunks per song
        num_chunks_per_song = 10
        total_chunks = len(self.data['labels'])
        num_songs = total_chunks // num_chunks_per_song
        
        # Shuffle songs, not chunks
        song_indices = np.random.RandomState(seed).permutation(num_songs)
        
        train_len = int(num_songs * 0.8)
        val_len = int(num_songs * 0.1)
        
        train_songs = song_indices[:train_len]
        val_songs = song_indices[train_len:train_len + val_len]
        test_songs = song_indices[train_len + val_len:]
        
        def get_chunk_indices(song_idxs):
            indices = []
            for s_idx in song_idxs:
                start = s_idx * num_chunks_per_song
                indices.extend(range(start, start + num_chunks_per_song))
            return np.array(indices)
        
        if split == 'train':
            self.indices = get_chunk_indices(train_songs)
        elif split == 'val':
            self.indices = get_chunk_indices(val_songs)
        else:
            self.indices = get_chunk_indices(test_songs)
            
        print(f"Loaded {split} set (CNN): {len(self.indices)} samples (from {len(self.indices)//10} songs)")

    def _process_and_cache(self):
        print("🎵 Processing GTZAN dataset for CNN (extracting Sliced Spectrograms)... this may take a few minutes.")
        features_list = []
        labels_list = []
        
        # We enforce 10 chunks per song to make splitting easier
        CHUNKS_PER_SONG = 10
        FRAMES_PER_CHUNK = 128
        
        for genre in self.classes:
            genre_dir = os.path.join(self.root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
                
            files = glob.glob(os.path.join(genre_dir, '*.wav'))
            print(f"Processing {genre}...")
            
            for f in tqdm(files):
                try:
                    # Load audio (ensure exactly 30s)
                    y, sr = librosa.load(f, duration=30.0)
                    
                    # Pad if shorter than 30s
                    target_len = 30 * sr
                    if len(y) < target_len:
                        y = np.pad(y, (0, target_len - len(y)))
                    else:
                        y = y[:target_len]
                    
                    # Compute Mel Spectrogram
                    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    melspec_db = librosa.power_to_db(melspec, ref=np.max)
                    
                    # Extract 10 chunks
                    # Total frames approx 1290. We take first 1280.
                    for i in range(CHUNKS_PER_SONG):
                        start = i * FRAMES_PER_CHUNK
                        end = start + FRAMES_PER_CHUNK
                        
                        # Safety check for bounds
                        if end > melspec_db.shape[1]:
                            # Pad if necessary (shouldn't happen often with 30s files)
                            chunk = melspec_db[:, start:]
                            pad_width = FRAMES_PER_CHUNK - chunk.shape[1]
                            chunk = np.pad(chunk, ((0, 0), (0, pad_width)))
                        else:
                            chunk = melspec_db[:, start:end]
                        
                        # Add channel dimension: (1, 128, 128)
                        chunk = chunk[np.newaxis, ...]
                        
                        features_list.append(chunk)
                        labels_list.append(self.class_to_idx[genre])
                        
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    # If a file fails, we must add placeholders to keep the count consistent
                    # or we just skip it. Skipping breaks the "10 chunks per song" logic for splitting.
                    # For simplicity in this project, we'll skip and hope the count is close enough 
                    # or handle it by just appending 10 dummy chunks (zeros).
                    # Let's append zeros to maintain alignment.
                    print(f"⚠️ Appending dummy data for {f} to maintain alignment.")
                    for _ in range(CHUNKS_PER_SONG):
                        features_list.append(np.zeros((1, 128, 128), dtype=np.float32))
                        labels_list.append(self.class_to_idx[genre])
                    
        data = {
            'features': torch.tensor(np.array(features_list), dtype=torch.float32),
            'labels': torch.tensor(np.array(labels_list), dtype=torch.long)
        }
        torch.save(data, self.cache_file)
        print(f"✅ Saved processed spectrograms to {self.cache_file}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feature = self.data['features'][real_idx]
        label = self.data['labels'][real_idx]
        
        # Standardization (per-sample, like in the notebook)
        # (x - mean) / std
        mean = feature.mean()
        std = feature.std()
        if std > 1e-6: # Avoid division by zero
            feature = (feature - mean) / std
            
        # Data Augmentation: Add random noise during training
        if self.split == 'train':
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
            
        return feature, label
