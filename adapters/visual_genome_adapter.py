"""Visual Genome dataset adapter - True lazy loading implementation."""

import os
import sys
import gc
import time
import random
from typing import List, Dict, Any, Optional, Iterator
from collections import OrderedDict
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class VisualGenomeAdapter(BaseDatasetAdapter):
    """Adapter for Visual Genome dataset with true lazy loading implementation.
    
    This implementation only loads sample indices and essential metadata upfront,
    then loads actual data on-demand during iteration. This dramatically reduces
    memory usage for large datasets.
    """

    def __init__(self, root_path: str, split: str = "train", transform=None, **kwargs):
        """
        Initialize Visual Genome adapter with true lazy loading.
        
        Args:
            root_path: Path for caching dataset
            split: Dataset split (only 'train' is available)
            transform: Image transformations to apply
            **kwargs: Additional arguments including:
                - max_samples: Maximum number of samples to use
                - min_objects: Minimum number of objects per image (default: 1)
                - max_objects: Maximum number of objects per image (default: 20)
                - use_synsets: Whether to use synsets instead of names (default: False)
                - config_name: Visual Genome config to use (default: "objects_v1.2.0")
        """
        self.min_objects = kwargs.get('min_objects', 1)
        self.max_objects = kwargs.get('max_objects', 20)
        self.use_synsets = kwargs.get('use_synsets', False)
        self.max_samples = kwargs.get('max_samples', None)
        self.config_name = kwargs.get('config_name', 'objects_v1.2.0')
        
        # True lazy loading: only store indices and minimal metadata
        self._dataset = None
        self._sample_indices = []
        self._classes_cache = None
        
        # Small image cache for frequently accessed images
        self._image_cache = OrderedDict()
        self._cache_size = 50  # Smaller cache for memory efficiency
        
        # Visual Genome only has train split
        if split != "train":
            print(f"Warning: Visual Genome only has 'train' split, ignoring split='{split}'")
        
        # Remove parameters that shouldn't be passed to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['min_objects', 'max_objects', 'use_synsets', 'max_samples', 'config_name']}
        
        super().__init__(root_path, transform=transform, split="train", **filtered_kwargs)

    def _clean_synset_label(self, synset: str) -> str:
        """
        Clean synset labels by removing WordNet suffixes to make them CLIP-friendly.
        
        Examples:
        - 'accent.n.01' -> 'accent'
        - 'air_conditioner.n.01' -> 'air conditioner'
        - 'adult.n.01' -> 'adult'
        
        Args:
            synset: Original synset string (e.g., 'accent.n.01')
            
        Returns:
            Cleaned label string (e.g., 'accent')
        """
        if not synset or not isinstance(synset, str):
            return synset
        
        # Remove WordNet suffixes like .n.01, .v.02, .a.01, .r.01
        # These suffixes indicate part of speech (n=noun, v=verb, a=adjective, r=adverb) and sense number
        parts = synset.split('.')
        if len(parts) >= 3:
            # Take only the word part before the first dot
            cleaned = parts[0]
        else:
            cleaned = synset
        
        # Replace underscores with spaces for better readability
        cleaned = cleaned.replace('_', ' ')
        
        return cleaned.strip()

    def _load_data(self, **kwargs):
        """Load Visual Genome data with true lazy loading - only indices and essential metadata."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Loading Visual Genome dataset ({self.config_name}) with true lazy loading...")
        print("Only pre-scanning for valid samples, actual data loaded on-demand.")
        
        # Set environment for better memory management
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
        os.environ['HF_DATASETS_CACHE'] = self.root_path
        
        try:
            # Load dataset without streaming for index access
            self._dataset = load_dataset(
                "visual_genome", 
                self.config_name,
                cache_dir=self.root_path,
                trust_remote_code=True,
                split='train'
            )
            
            # Get dataset size safely
            try:
                if hasattr(self._dataset, '__len__'):
                    try:
                        dataset_size = len(self._dataset)  # type: ignore[arg-type]
                        print(f"Dataset loaded. Total available samples: {dataset_size}")
                    except TypeError:
                        print("Dataset loaded. Using streaming mode.")
                        dataset_size = 50000  # Default estimate
                else:
                    print("Dataset loaded. Using streaming mode.")
                    dataset_size = 50000  # Default estimate
            except:
                print("Dataset loaded. Size unknown, using estimate.")
                dataset_size = 50000
            
        except Exception as e:
            print(f"Error loading Visual Genome dataset: {e}")
            # Try fallback version
            try:
                print("Trying with fallback version...")
                self._dataset = load_dataset(
                    "visual_genome", 
                    "objects_v1.0.0",
                    cache_dir=self.root_path,
                    trust_remote_code=True,
                    split='train'
                )
                self.config_name = "objects_v1.0.0"
                dataset_size = len(self._dataset) if hasattr(self._dataset, '__len__') else 50000  # type: ignore[arg-type]
                print("Successfully loaded fallback version")
            except Exception as e2:
                raise Exception(f"Failed to load Visual Genome dataset: {e2}")
        
        # Pre-scan to find valid sample indices (lightweight operation)
        print("Pre-scanning for valid samples...")
        valid_indices = []
        sample_count = 0
        
        # Determine scanning strategy based on max_samples
        if self.max_samples:
            # Limited samples: scan more to ensure we get enough valid ones
            max_scan = self.max_samples * 3
            scan_limit = min(dataset_size, max_scan)
            if scan_limit > 5000:
                # Random sampling for efficiency
                scan_indices = random.sample(range(min(dataset_size, 30000)), min(scan_limit, 5000))
            else:
                scan_indices = list(range(scan_limit))
            print(f"Target samples: {self.max_samples}, scanning {len(scan_indices)} samples to find valid data...")
        else:
            # No sample limit: use more comprehensive scanning
            max_scan = min(dataset_size, 30000)  # Increased limit for no max_samples
            if dataset_size > 30000:
                # For very large datasets, sample more broadly
                scan_indices = random.sample(range(dataset_size), 30000)
            else:
                # For smaller datasets, scan all or most
                scan_indices = list(range(min(dataset_size, max_scan)))
            print(f"No sample limit specified, scanning {len(scan_indices)} samples to find all valid data...")
        
        for scan_idx, actual_idx in enumerate(scan_indices):
            try:
                if self.max_samples is not None and len(valid_indices) >= self.max_samples:
                    break
                
                # Try to access the sample safely
                try:
                    example = self._dataset[actual_idx]  # type: ignore[index]
                except (IndexError, KeyError, TypeError, AttributeError):
                    continue
                
                # Extract objects safely
                if isinstance(example, dict):
                    objects = example.get("objects", [])
                else:
                    objects = getattr(example, 'objects', [])
                    
                if not objects:
                    continue
                    
                num_objects = len(objects)
                if num_objects < self.min_objects or (self.max_objects and num_objects > self.max_objects):
                    continue
                
                # Check if we can extract valid labels
                has_valid_labels = False
                for obj in objects[:3]:  # Check first few objects only
                    if self.use_synsets:
                        synsets = obj.get("synsets", []) if isinstance(obj, dict) else getattr(obj, 'synsets', [])
                        if synsets:
                            has_valid_labels = True
                            break
                    else:
                        names = obj.get("names", []) if isinstance(obj, dict) else getattr(obj, 'names', [])
                        if names:
                            has_valid_labels = True
                            break
                
                if has_valid_labels:
                    valid_indices.append(actual_idx)
                
                sample_count += 1
                if sample_count % 200 == 0:
                    print(f"Scanned {sample_count} samples, found {len(valid_indices)} valid samples...")
                    
            except Exception as e:
                if scan_idx < 3:  # Only show first few errors
                    print(f"Error checking sample {actual_idx}: {e}")
                continue
            
        if len(valid_indices) == 0:
            raise Exception("Failed to find any valid samples during pre-scanning")

        self._sample_indices = valid_indices
        print(f"✅ Found {len(valid_indices)} valid samples for lazy loading")
        
        # Return minimal metadata - just indices
        return [{'sample_index': idx} for idx in valid_indices]

    def _get_classes(self) -> List[str]:
        """Get all unique class names by sampling from the dataset."""
        if self._classes_cache is not None:
            return self._classes_cache
        
        print("Sampling dataset to discover class names...")
        all_labels = set()
        
        # Adaptive sampling based on dataset size
        if len(self._sample_indices) < 1000:
            # Small dataset: sample all
            sample_size = len(self._sample_indices)
        elif len(self._sample_indices) < 5000:
            # Medium dataset: sample 80%
            sample_size = int(len(self._sample_indices) * 0.8)
        else:
            # Large dataset: sample at least 2000 but up to 50%
            sample_size = min(max(2000, len(self._sample_indices) // 2), 10000)
        
        print(f"Sampling {sample_size} out of {len(self._sample_indices)} samples for class discovery...")
        
        # Sample random indices to get diverse class coverage
        sample_indices = random.sample(self._sample_indices, sample_size) if len(self._sample_indices) > sample_size else self._sample_indices
        
        for i, sample_idx in enumerate(sample_indices):
            try:
                if self._dataset is None:
                    break
                    
                example = self._dataset[sample_idx]  # type: ignore[index]
                
                # Extract objects safely
                if isinstance(example, dict):
                    objects = example.get("objects", [])
                else:
                    objects = getattr(example, 'objects', [])
                
                for obj in objects[:3]:  # Limit objects per image for efficiency
                    if self.use_synsets:
                        synsets = obj.get("synsets", []) if isinstance(obj, dict) else getattr(obj, 'synsets', [])
                        for synset in synsets[:2]:  # Limit synsets per object
                            if synset and isinstance(synset, str):
                                # Clean synset label to remove WordNet suffixes
                                cleaned_label = self._clean_synset_label(synset.strip())
                                all_labels.add(cleaned_label)
                    else:
                        names = obj.get("names", []) if isinstance(obj, dict) else getattr(obj, 'names', [])
                        for name in names[:2]:  # Limit names per object
                            if name and isinstance(name, str):
                                all_labels.add(name.strip())
                
                if i % 50 == 0 and i > 0:
                    print(f"Sampled {i} images, found {len(all_labels)} unique classes so far...")
                    
            except Exception as e:
                if i < 3:
                    print(f"Error sampling classes from index {sample_idx}: {e}")
                continue
        
        self._classes_cache = sorted(list(all_labels))
        print(f"Discovered {len(self._classes_cache)} unique classes from {len(sample_indices)} samples")
        
        return self._classes_cache

    def __getitem__(self, idx: int) -> tuple:
        """Get item by index with on-demand loading (lazy loading)."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        sample_index = item['sample_index']
        
        # Check cache first
        cache_key = f"sample_{sample_index}"
        if cache_key in self._image_cache:
            image, label = self._image_cache[cache_key]
            # Move to end (most recently used)
            self._image_cache.move_to_end(cache_key)
        else:
            # Load data on demand
            image, label = self._load_sample_on_demand(sample_index)
            
            # Add to cache with size limit
            self._image_cache[cache_key] = (image, label)
            if len(self._image_cache) > self._cache_size:
                # Remove least recently used
                self._image_cache.popitem(last=False)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index for consistency with base class
        if isinstance(label, str) and label in self.class_to_idx:
            label_idx = self.class_to_idx[label]
        else:
            label_idx = 0  # Default fallback
        
        return image, label_idx

    def _load_sample_on_demand(self, sample_index: int) -> tuple:
        """Load a specific sample on demand."""
        try:
            if self._dataset is None:
                raise Exception("Dataset not loaded")
            
            # Try direct indexing first, with fallback to iteration
            try:
                try:
                    example = self._dataset[sample_index]  # type: ignore[index]
                except (IndexError, KeyError, TypeError, AttributeError):
                    # Fallback for iterable datasets
                    example = None
                    for i, item in enumerate(self._dataset):
                        if i == sample_index:
                            example = item
                            break
                    if example is None:
                        raise IndexError(f"Sample {sample_index} not found")
            except (IndexError, TypeError):
                # Create fallback data
                return Image.new('RGB', (224, 224), color='gray'), "unknown"
            
            # Extract image
            if isinstance(example, dict):
                image = example.get("image")
                objects = example.get("objects", [])
            else:
                image = getattr(example, 'image', None)
                objects = getattr(example, 'objects', [])
            
            if image is None:
                # Create placeholder if image not available
                image = Image.new('RGB', (224, 224), color='gray')
            elif hasattr(image, 'convert'):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            # Extract primary label
            primary_label = "unknown"
            if objects:
                obj = objects[0]  # Use first object
                if self.use_synsets:
                    synsets = obj.get("synsets", []) if isinstance(obj, dict) else getattr(obj, 'synsets', [])
                    if synsets and synsets[0]:
                        # Clean synset label to remove WordNet suffixes
                        primary_label = self._clean_synset_label(synsets[0].strip())
                else:
                    names = obj.get("names", []) if isinstance(obj, dict) else getattr(obj, 'names', [])
                    if names and names[0]:
                        primary_label = names[0].strip()
            
            return image, primary_label
            
        except Exception as e:
            print(f"Error loading sample {sample_index}: {e}")
            # Return placeholder
            return Image.new('RGB', (224, 224), color='gray'), "unknown"

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
        sample_index = item['sample_index']
        
        try:
            if self._dataset is None:
                raise Exception("Dataset not loaded")
            
            # Try to get sample info safely  
            try:
                # Use duck typing to safely access the dataset
                example = None
                try:
                    # Try direct indexing with runtime error handling
                    example = self._dataset[sample_index]  # type: ignore[index]
                except (IndexError, KeyError, TypeError, AttributeError):
                    # Fallback for iterable datasets or failed indexing
                    for i, dataset_item in enumerate(self._dataset):
                        if i == sample_index:
                            example = dataset_item
                            break
                    if example is None:
                        raise IndexError(f"Sample {sample_index} not found")
            except (IndexError, TypeError):
                return {
                    'sample_index': sample_index,
                    'error': 'Could not access sample'
                }
            
            if isinstance(example, dict):
                return {
                    'sample_index': sample_index,
                    'image_id': example.get('image_id', f'img_{sample_index}'),
                    'url': example.get('url', 'N/A'),
                    'width': example.get('width', 0),
                    'height': example.get('height', 0),
                    'num_objects': len(example.get('objects', [])),
                    'coco_id': example.get('coco_id'),
                    'flickr_id': example.get('flickr_id')
                }
            else:
                return {
                    'sample_index': sample_index,
                    'image_id': getattr(example, 'image_id', f'img_{sample_index}'),
                    'url': getattr(example, 'url', 'N/A'),
                    'width': getattr(example, 'width', 0),
                    'height': getattr(example, 'height', 0),
                    'num_objects': len(getattr(example, 'objects', [])),
                    'coco_id': getattr(example, 'coco_id', None),
                    'flickr_id': getattr(example, 'flickr_id', None)
                }
        except Exception as e:
            return {
                'sample_index': sample_index,
                'error': str(e)
            }
