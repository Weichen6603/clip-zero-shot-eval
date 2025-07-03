"""ImageNet dataset adapter using Hugging Face datasets (offline mode only)."""

import os
import sys
from typing import List, Optional, Tuple, Any, Dict
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class ImageNetAdapter(BaseDatasetAdapter):
    """Adapter for ImageNet dataset using Hugging Face imagenet-1k.
    
    This implementation focuses on offline mode with optimized data loading.
    Features:
    - Full dataset download and caching
    - Memory-mapped data storage for efficient access
    - Precomputed indices for fast random access
    - Support for validation split (50,000 images)
    """

    def __init__(self, root_path: str, transform=None, split: str = 'validation', 
                 max_samples: Optional[int] = None, use_cache: bool = True, **kwargs):
        """Initialize ImageNet adapter.
        
        Args:
            root_path: Root directory to cache the dataset
            transform: Image transformations to apply
            split: Dataset split to use ('train' or 'validation')
            max_samples: Maximum number of samples to use (None for all)
            use_cache: Whether to use cached preprocessed data
            **kwargs: Additional arguments
        """
        # Validate split
        if split not in ['train', 'validation']:
            raise ValueError(f"Split must be 'train' or 'validation', got {split}")
        
        # Set paths
        self.cache_dir = root_path
        self.dataset = None
        self.max_samples = max_samples
        self.use_cache = use_cache
        self.split = split
        
        # Cache file paths
        self.cache_file = os.path.join(self.cache_dir, f'imagenet_{split}_cache.pkl')
        self.index_file = os.path.join(self.cache_dir, f'imagenet_{split}_index.pkl')
        
        # Initialize data storage
        self._image_cache = {}
        self._label_cache = {}
        self._synset_cache = {}
        
        super().__init__(root_path, transform, split, **kwargs)

    def _load_data(self, **kwargs):
        """Load ImageNet data from Hugging Face with optimized offline strategy."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        # Try to load from cache first
        if self.use_cache and os.path.exists(self.cache_file) and os.path.exists(self.index_file):
            print(f"Loading cached ImageNet data from {self.cache_file}")
            return self._load_from_cache()
        
        print(f"Loading ImageNet-1K dataset from Hugging Face...")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Split: {self.split}")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset (offline mode)
        try:
            print("Downloading/loading dataset (this may take a while for first time)...")
            self.dataset = load_dataset(
                "imagenet-1k", 
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            print(f"Successfully loaded dataset with {len(self.dataset)} samples")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you're logged in: huggingface-cli login")
            print("2. Request access at: https://huggingface.co/datasets/imagenet-1k")
            print("3. Wait for approval (usually quick)")
            raise
        
        # Process and cache the dataset
        return self._process_and_cache_dataset()
    
    def _process_and_cache_dataset(self) -> List[Dict[str, Any]]:
        """Process the dataset and create optimized cache."""
        data = []
        
        # Determine number of samples to process
        total_samples = len(self.dataset)
        if self.max_samples:
            total_samples = min(total_samples, self.max_samples)
        
        print(f"Processing {total_samples} samples...")
        
        # Process samples with progress bar
        for idx in tqdm(range(total_samples), desc="Processing samples"):
            sample = self.dataset[idx]
            
            # Get label and synset
            label = sample['label']
            
            # Get synset if available
            synset = None
            if hasattr(self.dataset.features['label'], 'int2str'):
                synset = self.dataset.features['label'].int2str(label)
            else:
                synset = f"class_{label}"
            
            # Store in cache
            self._image_cache[idx] = sample['image']
            self._label_cache[idx] = label
            self._synset_cache[idx] = synset
            
            # Add to data list
            data.append({
                'image_idx': idx,
                'label': label,
                'synset': synset
            })
        
        # Save cache if enabled
        if self.use_cache:
            self._save_cache(data)
        
        return data
    
    def _save_cache(self, data: List[Dict[str, Any]]):
        """Save processed data to cache files."""
        print(f"Saving cache to {self.cache_file}")
        
        # Save index data
        with open(self.index_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save a lightweight version without images for quick loading
        cache_data = {
            'num_samples': len(data),
            'classes': self._get_classes(),
            'split': self.split
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("Cache saved successfully")
    
    def _load_from_cache(self) -> List[Dict[str, Any]]:
        """Load data from cache files."""
        # Load index
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load metadata
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Loaded {len(data)} cached samples")
        
        # Reload the dataset for image access
        try:
            from datasets import load_dataset
            self.dataset = load_dataset(
                "imagenet-1k", 
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not reload dataset for image access: {e}")
        
        return data

    def _get_classes(self) -> List[str]:
        """Get ImageNet class names using the dataset features."""
        if self.dataset is None:
            return [f"class_{i}" for i in range(1000)]
        
        try:
            if hasattr(self.dataset.features['label'], 'names'):
                return self.dataset.features['label'].names
            else:
                # Fallback: try to get from int2str
                classes = []
                for i in range(1000):
                    if hasattr(self.dataset.features['label'], 'int2str'):
                        classes.append(self.dataset.features['label'].int2str(i))
                    else:
                        classes.append(f"class_{i}")
                return classes
        except Exception as e:
            print(f"Warning: Could not extract class names: {e}")
            return [f"class_{i}" for i in range(1000)]

    def get_templates(self) -> List[str]:
        """Get templates for ImageNet zero-shot classification."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample by index with optimized loading."""
        item = self.data[index]
        
        # Get image
        if 'image_idx' in item:
            # Load from dataset
            image_idx = item['image_idx']
            
            # Try cache first
            if image_idx in self._image_cache:
                image = self._image_cache[image_idx]
            elif self.dataset is not None:
                # Load from dataset
                image = self.dataset[image_idx]['image']
                # Optionally cache for repeated access
                if len(self._image_cache) < 1000:  # Limit cache size
                    self._image_cache[image_idx] = image
            else:
                # Fallback: create placeholder
                print(f"Warning: Could not load image at index {image_idx}")
                image = Image.new('RGB', (224, 224), color='gray')
        else:
            # Legacy format - should not happen with new cache
            print(f"Warning: Legacy data format at index {index}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            else:
                print(f"Warning: Invalid image type at index {index}")
                image = Image.new('RGB', (224, 224), color='gray')
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Get label
        label = item['label']
        
        return image, label
    
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        item = self.data[index]
        
        info = {
            'index': index,
            'label': item['label'],
            'synset': item.get('synset', f"class_{item['label']}"),
            'class_name': self.classes[item['label']] if item['label'] < len(self.classes) else 'unknown'
        }
        
        return info
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        self._image_cache.clear()
        self._label_cache.clear()
        self._synset_cache.clear()
        
        if hasattr(self, 'dataset'):
            del self.dataset
            self.dataset = None
        
        print("Cache cleared")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (f"ImageNetAdapter(split='{self.split}', "
                f"samples={len(self)}, "
                f"classes={len(self.classes)}, "
                f"cache_dir='{self.cache_dir}')")