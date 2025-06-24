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

    def __init__(self, root_path: str, transform=None, split: str = 'validation', **kwargs):
        """Initialize ImageNet adapter.
        
        Args:
            root_path: Root directory to cache the dataset (will use as cache_dir)
            transform: Image transformations to apply
            split: Dataset split to use ('validation' for ImageNet-1K)
            **kwargs: Additional arguments
        """
        # Set cache directory to the specified path
        self.cache_dir = root_path
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
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset from Hugging Face
        try:
            # Note: For official imagenet-1k dataset, authentication is handled automatically
            self.dataset = load_dataset(
                "imagenet-1k", 
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True  # Required for custom dataset code
            )
            print(f"Successfully loaded dataset")
            print(f"Dataset info: {self.dataset}")
            print(f"Dataset features: {self.dataset.features}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: You may need to authenticate with Hugging Face:")
            print("1. Install huggingface_hub: pip install huggingface_hub")
            print("2. Login: huggingface-cli login")
            print("3. Request access to imagenet-1k dataset if needed")
            raise
        
        # Convert to our format
        data = []
        print("Processing dataset samples...")
        for idx, sample in enumerate(self.dataset):
            if idx >= 1000:  # Limit for testing, remove this line for full dataset
                break
            
            # Get synset string using int2str method
            synset = None
            try:
                synset = self.dataset.features['label'].int2str(sample['label'])
            except:
                synset = f"synset_{sample['label']}"
            
            # Save image temporarily to match base class interface
            import tempfile
            import os
            temp_dir = os.path.join(self.cache_dir, "temp_images")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"img_{idx}.jpg")
            
            # Save PIL image to temporary file
            sample['image'].save(temp_path, "JPEG")
            
            data.append({
                'image_path': temp_path,  # Path to temporary image file for base class
                'label': sample['label'],  # This is the class index (0-999)
                'synset': synset,  # This is the synset string (e.g., 'n01440764')
                'image_obj': sample['image']  # Keep original PIL object for reference
            })
            if idx % 1000 == 0:
                print(f"Processed {idx} samples...")
        
        print(f"Successfully processed {len(data)} samples")
        
        return data

    def _get_classes(self) -> List[str]:
        """Get ImageNet class names using the full 1000 classes from the dataset features."""
        if self.dataset is None:
            return []
        
        try:
            # Use the dataset features to get all 1000 class names
            # This ensures we have the complete mapping even if we only process a subset of samples
            label_feature = self.dataset.features['label']
            if hasattr(label_feature, 'names'):
                # This should give us all 1000 ImageNet class names
                return label_feature.names
            else:
                # Fallback: This shouldn't happen with the official dataset
                print("Warning: Could not get class names from dataset features")
                return [f"class_{i}" for i in range(1000)]
                
        except Exception as e:
            print(f"Warning: Could not extract class names from features: {e}")
            # Fallback: create 1000 generic class names
            return [f"class_{i}" for i in range(1000)]

    def get_templates(self) -> List[str]:
        """Templates for ImageNet zero-shot classification."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()
