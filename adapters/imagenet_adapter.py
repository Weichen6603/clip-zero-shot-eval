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
                 streaming: bool = True, **kwargs):
        """Initialize ImageNet adapter.
        
        Args:
            root_path: Root directory to cache the dataset (will use as cache_dir)
            transform: Image transformations to apply
            split: Dataset split to use ('validation' for ImageNet-1K)
            streaming: Whether to use streaming mode (True) or download full dataset (False)
            **kwargs: Additional arguments
        """
        # Set cache directory to the specified path
        self.cache_dir = root_path
        self.dataset = None  # Will be loaded in _load_data
        self.max_samples = kwargs.get("max_samples", None)
        
        # streaming config: explicit arg > kwargs > default True
        if 'streaming' in kwargs:
            self.streaming = kwargs['streaming']
        else:
            self.streaming = streaming

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
        print(f"Streaming mode: {self.streaming}")
        
        # Create cache directory if it doesn't exist
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset from Hugging Face
        try:
            # Note: For official imagenet-1k dataset, authentication is handled automatically
            self.dataset = load_dataset(
                "imagenet-1k", 
                split=self.split,
                streaming=self.streaming,  # Use streaming parameter from config
                cache_dir=self.cache_dir,
                trust_remote_code=True  # Required for custom dataset code
            )
            print(f"Successfully loaded dataset")
            print(f"Dataset info: {self.dataset}")
            print(f"Dataset type: {type(self.dataset)}")
            # Only print features if dataset is not a dict (i.e., is a Dataset or IterableDataset)
            if not isinstance(self.dataset, dict):
                if hasattr(self.dataset, 'features'):
                    print(f"Dataset features: {self.dataset.features}")
                else:
                    print("Dataset features: Not available (streaming mode)")
            else:
                print(f"Dataset is a dict type ({type(self.dataset)}), skipping features print.")
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
        try:
            from tqdm import tqdm
        except ImportError:
            print("tqdm not installed. Install it with: pip install tqdm")
            tqdm = lambda x, **kwargs: x  # fallback: no progress bar
        
        total = self.max_samples if self.max_samples is not None else None
        # Only try to get length for non-streaming datasets that support it
        if not self.streaming:
            try:
                # Import Dataset type to check instance type
                from datasets import Dataset
                if isinstance(self.dataset, Dataset):
                    dataset_len = len(self.dataset)
                    total = min(dataset_len, self.max_samples or dataset_len)
            except:
                pass  # Use max_samples as total
        
        for idx, sample in enumerate(tqdm(self.dataset, total=total, desc="Processing samples")):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            
            # Get synset string using int2str method
            synset = None
            try:
                if self.dataset is not None and not isinstance(self.dataset, dict):
                    features = getattr(self.dataset, 'features', None)
                    if features is not None and 'label' in features:
                        synset = features['label'].int2str(sample['label'])
                    else:
                        synset = f"synset_{sample['label']}"
                else:
                    synset = f"synset_{sample['label']}"
            except Exception as e:
                synset = f"synset_{sample['label']}"
            
            if self.streaming:
                # In streaming mode, keep PIL image object directly to avoid file I/O
                data.append({
                    'image_path': idx,  # Use index as identifier
                    'label': sample['label'],  # This is the class index (0-999)
                    'synset': synset,  # This is the synset string (e.g., 'n01440764')
                    'image_obj': sample['image']  # Keep original PIL object
                })
            else:
                # In offline mode, save image temporarily to match base class interface
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
        
        print(f"Successfully processed {len(data)} samples")
        
        return data

    def _get_classes(self) -> List[str]:
        """Get ImageNet class names using the full 1000 classes from the dataset features."""
        if self.dataset is None:
            return []
        try:
            # Only access features if dataset is not a dict and features is not None
            if not isinstance(self.dataset, dict):
                features = getattr(self.dataset, 'features', None)
                if features is not None and 'label' in features:
                    label_feature = features['label']
                    if hasattr(label_feature, 'names'):
                        return label_feature.names
                    else:
                        print("Warning: Could not get class names from dataset features")
                        return [f"class_{i}" for i in range(1000)]
                else:
                    print(f"Warning: features is None or missing 'label', cannot extract class names.")
                    return [f"class_{i}" for i in range(1000)]
            else:
                print(f"Warning: self.dataset is a dict type ({type(self.dataset)}), cannot extract class names from features.")
                return [f"class_{i}" for i in range(1000)]
        except Exception as e:
            print(f"Warning: Could not extract class names from features: {e}")
            return [f"class_{i}" for i in range(1000)]

    def get_templates(self) -> List[str]:
        """Templates for ImageNet zero-shot classification."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample by index, handling both streaming and offline modes."""
        item = self.data[index]
        
        if self.streaming:
            # In streaming mode, use the PIL image object directly
            image = item['image_obj']
        else:
            # In offline mode, load from temporary file path
            image_path = item['image_path']
            if 'image_obj' in item:
                # Use cached PIL object if available
                image = item['image_obj']
            else:
                # Load from file path
                image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Convert PIL image to tensor if no transform is provided
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Get label
        label = item['label']
        
        return image, label
