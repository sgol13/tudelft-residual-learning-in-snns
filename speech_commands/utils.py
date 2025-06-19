import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Tuple, Optional
import yaml
import logging
from pathlib import Path
import time


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str) -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


class SpeechCommandsDataset(Dataset):
    def __init__(self, split: str, config: dict, transform=None):
        """
        Initialize the Speech Commands dataset.
        
        Args:
            split: 'train', 'validation', or 'test'
            config: Configuration dictionary
            transform: Optional transform to apply to the audio
        """
        self.config = config
        self.transform = transform
        
        # Load dataset from HuggingFace with trust_remote_code=True and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.dataset = load_dataset(
                    "google/speech_commands", 
                    "v0.02", 
                    split=split,
                    trust_remote_code=True,
                    # streaming=True  # Use streaming to avoid downloading the entire dataset
                )
                # Convert streaming dataset to regular dataset with a subset
                # self.dataset = list(self.dataset.take(500))  # Take only 1000 samples for testing
                self.dataset = self.dataset.filter(lambda x: x['label'] in range(8))
                
                # Print available classes
                print("\nAvailable classes in the dataset:")
                classes = sorted(set(item['label'] for item in self.dataset))
                print(classes)
                print(f"Total number of classes: {len(classes)}\n")
                
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                time.sleep(5)
        
        # Setup audio processing
        self.sample_rate = config['data']['sample_rate']
        self.duration = config['data']['duration']
        self.num_steps = config['training']['num_steps']
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(item['label'] for item in self.dataset)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        
        # Load and preprocess audio
        waveform = torch.tensor(item['audio']['array'], dtype=torch.float32)
        waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Resample if necessary
        if item['audio']['sampling_rate'] != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                item['audio']['sampling_rate'], 
                self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Pad or trim to desired duration
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Convert to spikes (simple threshold-based encoding)
        spikes = (waveform > 0.1).float()
        
        # Repeat for number of time steps
        spikes = spikes.repeat(self.num_steps, 1, 1)
        
        # Get label
        label = self.label_to_idx[item['label']]
        
        return spikes, label


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = SpeechCommandsDataset('train', config)
    val_dataset = SpeechCommandsDataset('validation', config)
    test_dataset = SpeechCommandsDataset('test', config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Reduced workers for testing
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Reduced workers for testing
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Reduced workers for testing
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, save_path: str) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   checkpoint_path: str) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'] 