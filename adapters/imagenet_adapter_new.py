"""ImageNet dataset adapter using Hugging Face datasets."""

import os
import sys
from typing import List, Optional, Tuple, Any
from PIL import Image
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class ImageNetAdapter(BaseDatasetAdapter):
    """Adapter for ImageNet dataset using Hugging Face mlx-vision/imagenet-1k."""

    def __init__(self, root_path: str, transform=None, split: str = 'validation', 
                 use_auth_token: bool = True, **kwargs):
        """Initialize ImageNet adapter.
        
        Args:
            root_path: Root directory to cache the dataset (will use as cache_dir)
            transform: Image transformations to apply
            split: Dataset split to use ('validation' for ImageNet-1K)
            use_auth_token: Whether to use HuggingFace auth token
            **kwargs: Additional arguments
        """
        # Set cache directory to the specified path
        self.cache_dir = root_path
        self.use_auth_token = use_auth_token
        self.dataset = None  # Will be loaded in _load_data
        
        super().__init__(root_path, transform, split, **kwargs)

    def _load_data(self, **kwargs):
        """Load ImageNet data from Hugging Face.

        Downloads and caches the dataset at the specified Windows path via WSL mount.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading ImageNet-1K dataset from Hugging Face...")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Split: {self.split}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset from Hugging Face
        try:
            self.dataset = load_dataset(
                "mlx-vision/imagenet-1k", 
                split=self.split,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token
            )
            print(f"Successfully loaded {len(self.dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: You may need to authenticate with Hugging Face:")
            print("1. Install huggingface_hub: pip install huggingface_hub")
            print("2. Login: huggingface-cli login")
            raise
        
        # Convert to our format
        data = []
        for idx, sample in enumerate(self.dataset):
            data.append({
                'image_idx': idx,  # Store index instead of path since images are in memory
                'label': sample['label'],  # This should be the class index
                'image_obj': sample['image']  # Store PIL image object
            })
        
        return data

    def _get_classes(self) -> List[str]:
        """Get ImageNet class names."""
        if self.dataset is None:
            return []
        
        # Get class names from the dataset features
        try:
            # The dataset should have a ClassLabel feature for labels
            label_feature = self.dataset.features['label']
            if hasattr(label_feature, 'names'):
                return label_feature.names
            else:
                # Fallback: extract unique labels and create generic names
                unique_labels = sorted(set(item['label'] for item in self.data))
                return [f"class_{i}" for i in unique_labels]
        except Exception as e:
            print(f"Warning: Could not extract class names: {e}")
            # Fallback: create generic class names
            unique_labels = sorted(set(item['label'] for item in self.data))
            return [f"class_{i}" for i in unique_labels]

    def get_templates(self) -> List[str]:
        """Templates for ImageNet zero-shot classification."""
        # Import here to avoid circular import
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Returns:
            image: Transformed image tensor
            label: Integer class label
        """
        item = self.data[idx]
        
        # Get image from stored PIL object
        image = item['image_obj']
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Label should already be an integer
        label = item['label']
        if not isinstance(label, int):
            raise ValueError(f"Expected integer label, got {type(label)}")
        
        return image, label
