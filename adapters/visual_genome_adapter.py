"""Visual Genome dataset adapter - Ultra memory optimized version."""

import os
import sys
import gc
import time
from typing import List, Dict, Any
from collections import OrderedDict
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class VisualGenomeAdapter(BaseDatasetAdapter):
    """Adapter for Visual Genome dataset with ultra memory optimization."""

    def __init__(self, root_path: str, split: str = "train", transform=None, **kwargs):
        """
        Initialize Visual Genome adapter with ultra memory optimization.
        
        Args:
            root_path: Path for caching dataset
            split: Dataset split (only 'train' is available)
            transform: Image transformations to apply
            **kwargs: Additional arguments including:
                - max_samples: Maximum number of samples to use
                - min_objects: Minimum number of objects per image (default: 1)
                - max_objects: Maximum number of objects per image (default: None)
                - use_synsets: Whether to use synsets instead of names (default: False)
                - config_name: Visual Genome config to use (default: "objects_v1.2.0")
        """
        self.min_objects = kwargs.get('min_objects', 1)
        self.max_objects = kwargs.get('max_objects', None)
        self.use_synsets = kwargs.get('use_synsets', False)
        self.max_samples = kwargs.get('max_samples', None)
        self.config_name = kwargs.get('config_name', 'objects_v1.2.0')
        
        # Ultra minimal image cache - only keep 100 images
        self._image_cache = OrderedDict()
        self._cache_size = 100
        
        # Store dataset reference for lazy loading
        self._dataset = None
        self._dataset_iter = None
        
        # Visual Genome only has train split
        if split != "train":
            print(f"Warning: Visual Genome only has 'train' split, ignoring split='{split}'")
        
        # Remove parameters that shouldn't be passed to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['min_objects', 'max_objects', 'use_synsets', 'max_samples', 'config_name']}
        
        super().__init__(root_path, transform=transform, split="train", **filtered_kwargs)

    def _load_data(self, **kwargs):
        """Load Visual Genome data with ultra memory optimization."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Loading Visual Genome dataset ({self.config_name}) with ultra memory optimization...")
        print("Only storing minimal metadata, images loaded on-demand.")
        
        # Set environment for better memory management
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
        os.environ['HF_DATASETS_CACHE'] = self.root_path
        
        try:
            # Load with streaming enabled for minimal memory footprint
            dataset = load_dataset(
                "visual_genome", 
                self.config_name,
                cache_dir=self.root_path,
                trust_remote_code=True,
                streaming=True,
                split='train'
            )
            
            if hasattr(dataset, 'shuffle'):
                dataset = dataset.shuffle(seed=42, buffer_size=1000)
            
        except Exception as e:
            print(f"Error loading Visual Genome dataset: {e}")
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                print("❌ Network connection issue. Please check internet connectivity.")
                raise Exception(f"Network error loading Visual Genome: {e}")
            
            # Fallback to older version
            print("Trying with fallback version...")
            try:
                dataset = load_dataset(
                    "visual_genome", 
                    "objects_v1.0.0",
                    cache_dir=self.root_path,
                    trust_remote_code=True,
                    streaming=True,
                    split='train'
                )
                self.config_name = "objects_v1.0.0"
            except Exception as e2:
                raise Exception(f"Failed to load Visual Genome dataset: {e2}")
        
        # Store dataset reference for lazy loading
        self._dataset = dataset
        
        # Only load metadata, not actual data
        metadata = []
        processed_count = 0
        
        print("Processing metadata (ultra-lite mode)...")
        
        try:
            # Use take() to limit samples early in streaming
            if self.max_samples:
                dataset_iter = dataset.take(self.max_samples * 2)  # Take more to account for filtering
            else:
                dataset_iter = dataset
            
            for idx, example in enumerate(dataset_iter):
                try:
                    # Early break if we have enough samples
                    if self.max_samples and processed_count >= self.max_samples:
                        break
                    
                    # Extract and validate objects
                    objects = example.get("objects", [])
                    if not objects:
                        continue
                        
                    num_objects = len(objects)
                    if num_objects < self.min_objects:
                        continue
                    if self.max_objects and num_objects > self.max_objects:
                        continue
                    
                    # Extract minimal label information
                    labels = []
                    for obj in objects[:5]:  # Only process first 5 objects to save memory
                        if self.use_synsets:
                            obj_synsets = obj.get("synsets", [])
                            if obj_synsets:
                                labels.extend(obj_synsets[:2])  # Max 2 synsets per object
                        else:
                            obj_names = obj.get("names", [])
                            if obj_names:
                                labels.extend(obj_names[:2])  # Max 2 names per object
                    
                    # Clean labels and limit count
                    labels = [label.strip() for label in set(labels) 
                             if label and isinstance(label, str) and len(label.strip()) > 0][:5]
                    
                    if not labels:
                        continue
                    
                    # Store absolute minimal metadata
                    metadata.append({
                        'idx': idx,
                        'label': labels[0],  # Primary label only
                        'image_id': example.get("image_id", f"img_{idx}"),
                        'url': example.get("url", "")[:100],  # Truncate URL to save memory
                        # Remove width, height, all_labels to save memory
                    })
                    
                    processed_count += 1
                    
                    # More frequent progress updates and garbage collection
                    if processed_count % 500 == 0:
                        print(f"Processed {processed_count} samples...")
                        gc.collect()  # Force garbage collection
                        
                except Exception as e:
                    if idx < 3:  # Show fewer error messages
                        print(f"Error processing sample {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error iterating through dataset: {e}")
            if len(metadata) == 0:
                raise Exception(f"Failed to load any samples: {e}")

        print(f"✅ Loaded {len(metadata)} Visual Genome samples (ultra-lite metadata)")
        if metadata:
            print(f"Sample labels: {[item['label'] for item in metadata[:3]]}")
        
        # Force garbage collection
        gc.collect()
        
        return metadata

    def _get_classes(self) -> List[str]:
        """Get all unique class names from the dataset."""
        if not hasattr(self, '_classes'):
            # Only use primary labels to reduce memory
            all_labels = {item['label'] for item in self.data}
            self._classes = sorted(list(all_labels))
            print(f"Found {len(self._classes)} unique classes")
        return self._classes

    def __getitem__(self, idx: int):
        """Get item by index with on-demand image loading."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        
        # Check cache first
        cache_key = item['image_id']
        if cache_key in self._image_cache:
            image = self._image_cache[cache_key]
            # Move to end (most recently used)
            self._image_cache.move_to_end(cache_key)
        else:
            # Load image on demand
            image = self._load_image_on_demand(item)
            
            # Add to cache with size limit
            self._image_cache[cache_key] = image
            if len(self._image_cache) > self._cache_size:
                # Remove least recently used
                self._image_cache.popitem(last=False)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

    def _load_image_on_demand(self, item: Dict[str, Any]) -> Image.Image:
        """Load image on demand from URL or dataset."""
        try:
            # Try to load from URL first (faster if available)
            url = item.get('url', '')
            if url and url.startswith('http'):
                try:
                    import requests
                    from io import BytesIO
                    
                    response = requests.get(url, timeout=10, stream=True)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        return image
                except Exception:
                    pass  # Fall back to dataset loading
            
            # Fallback: load from dataset (slower but more reliable)
            if self._dataset is None:
                raise Exception("Dataset not loaded")
            
            # This is expensive but necessary for reliability
            dataset_iter = iter(self._dataset)
            target_idx = item['idx']
            
            for i, example in enumerate(dataset_iter):
                if i == target_idx:
                    image = example.get('image')
                    if image and hasattr(image, 'convert'):
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        return image
                    break
            
            # If we can't load the image, create a placeholder
            print(f"Warning: Could not load image for item {item['image_id']}, using placeholder")
            return Image.new('RGB', (224, 224), color='gray')
            
        except Exception as e:
            print(f"Error loading image {item['image_id']}: {e}")
            # Return a placeholder image
            return Image.new('RGB', (224, 224), color='gray')

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def get_sample_image_label(self, idx: int = 0):
        """Get a sample image and label for testing."""
        if len(self.data) == 0:
            return None, None
        
        idx = min(idx, len(self.data) - 1)
        return self.__getitem__(idx)

    def get_templates(self) -> List[str]:
        """Templates for Visual Genome zero-shot classification."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get additional information about a sample."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")
        
        item = self.data[idx]
        return {
            'image_id': item['image_id'],
            'primary_label': item['label'],
            'all_labels': item.get('all_labels', [item['label']]),
            'num_objects': item.get('num_objects', 1),
            'image_size': item.get('image_size', (0, 0)),
            'url': item.get('url', 'N/A'),
            'coco_id': item.get('coco_id'),
            'flickr_id': item.get('flickr_id')
        }
