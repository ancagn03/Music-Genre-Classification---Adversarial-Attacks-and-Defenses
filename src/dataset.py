"""
Dataset loading and processing logic.
Handles loading MSD features and GTZAN audio/spectrograms.
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
    Checks if the dataset exists. If not, and running on Colab, downloads it.
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
    def __init__(self, root_dir='data/gtzan-music-genre-dataset', split='train', seed=42):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            split (str): 'train', 'val', or 'test'.
            seed (int): Random seed for reproducible splits.
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
        self.cache_file = os.path.join(root_dir, 'features_cnn.pt')
        
        download_gtzan_if_missing(root_dir)
        
        if not os.path.exists(self.cache_file):
            self._process_and_cache()
            
        self.data = torch.load(self.cache_file)
        
        # Splits
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
            
        print(f"Loaded {split} set (CNN): {len(self.indices)} samples")

    def _process_and_cache(self):
        print("🎵 Processing GTZAN dataset for CNN (extracting Spectrograms)... this may take a few minutes.")
        features_list = []
        labels_list = []
        
        target_size = (128, 128) # (n_mels, time_steps)
        
        for genre in self.classes:
            genre_dir = os.path.join(self.root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
                
            files = glob.glob(os.path.join(genre_dir, '*.wav'))
            print(f"Processing {genre}...")
            
            for f in tqdm(files):
                try:
                    # Load audio
                    y, sr = librosa.load(f, duration=30)
                    
                    # Compute Mel Spectrogram
                    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    melspec_db = librosa.power_to_db(melspec, ref=np.max)
                    
                    # Resize/Crop to 128x128
                    # We take the center 128 frames (approx 3 seconds)
                    # If shorter, we pad. If longer, we crop.
                    n_frames = melspec_db.shape[1]
                    if n_frames > 128:
                        start = (n_frames - 128) // 2
                        melspec_db = melspec_db[:, start:start+128]
                    else:
                        pad_width = 128 - n_frames
                        melspec_db = np.pad(melspec_db, ((0, 0), (0, pad_width)))
                    
                    # Add channel dimension: (1, 128, 128)
                    melspec_db = melspec_db[np.newaxis, ...]
                    
                    features_list.append(melspec_db)
                    labels_list.append(self.class_to_idx[genre])
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    
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
