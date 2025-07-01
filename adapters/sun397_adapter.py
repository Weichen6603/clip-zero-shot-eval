"""SUN397 dataset adapter."""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class SUN397Adapter(BaseDatasetAdapter):
    """Adapter for SUN397 dataset using Hugging Face Datasets."""

    def _load_data(self, use_huggingface=True, **kwargs):
        """Load SUN397 data using Hugging Face Datasets.
        
        Args:
            use_huggingface: Use Hugging Face datasets (recommended)
        """
        if use_huggingface:
            return self._load_data_huggingface(**kwargs)
        else:
            return self._load_data_tensorflow(**kwargs)
    
    def _load_data_huggingface(self, test_split_ratio=0.2, streaming=True, **kwargs):
        """Load SUN397 data using Hugging Face Datasets.
        
        Args:
            test_split_ratio: Ratio to split train data into train/test (default: 0.2)
            streaming: Enable streaming mode for Hugging Face datasets (default: True)
        """
        try:
            from datasets import load_dataset
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Hugging Face datasets is required for SUN397 dataset. "
                "Install it with: pip install datasets"
            ) from e

        try:
            print("Loading SUN397 dataset from Hugging Face...")
            
            # Handle different split requests
            if self.split == 'test':
                print("Note: SUN397 only has 'train' split. Creating test split from training data...")
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path, streaming=streaming)
                
                if not streaming:
                    train_test_split = dataset.train_test_split(test_size=test_split_ratio, shuffle=True, seed=42)
                    dataset = train_test_split['test']
                    print(f"Using test split ({len(dataset)} samples) from training data")
                else:
                    print("Streaming mode enabled, test split will be handled dynamically.")
            elif self.split == 'train':
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path, streaming=streaming)
                
                if not streaming and test_split_ratio > 0:
                    train_test_split = dataset.train_test_split(test_size=test_split_ratio, shuffle=True, seed=42)
                    dataset = train_test_split['train']
                    print(f"Using train split ({len(dataset)} samples) from training data")
                else:
                    print(f"Using full dataset ({'streaming mode' if streaming else len(dataset)} samples)")
            else:
                print(f"Warning: Split '{self.split}' not available. Using 'train' split.")
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path, streaming=streaming)

            max_samples = kwargs.get('max_samples', None)
            
            if streaming:
                # Streaming mode: Create lightweight metadata for lazy loading (即用即扔)
                print("Streaming mode enabled - creating lightweight metadata...")
                if max_samples is not None:
                    actual_length = max_samples
                    print(f"Limited to {max_samples} samples for testing")
                else:
                    # For streaming, we can't know exact length, use default estimate
                    actual_length = 19850  # SUN397 approximate size
                    print(f"Using estimated dataset size ({actual_length} samples)")
                
                # Store dataset reference for on-demand loading
                self._hf_dataset = dataset
                self._dataset_for_streaming = dataset
                
                # Create minimal metadata - no image objects stored (即用即扔)
                data = []
                for idx in range(actual_length):
                    data.append({
                        'image_path': idx,  # Index for lazy loading
                        'label': None,  # Will be loaded on-demand
                    })
                    
                    if (idx + 1) % 10000 == 0:
                        print(f"Created metadata for {idx + 1} samples...")
                
                print(f"✓ Successfully created lightweight metadata for {len(data)} samples")
                return data
            else:
                # Non-streaming mode: Original implementation
                dataset_length = len(dataset) if hasattr(dataset, '__len__') else 87003
                if max_samples is not None and max_samples < dataset_length:
                    print(f"Limited to {max_samples} samples for testing")
                    actual_length = max_samples
                else:
                    print(f"Using full dataset ({dataset_length} samples)")
                    actual_length = dataset_length

                self._hf_dataset = dataset
                data = []
                print(f"Creating metadata for {actual_length} samples...")
                for idx in range(actual_length):
                    data.append({
                        'image_path': idx,
                        'label': None,
                    })
                    if (idx + 1) % 10000 == 0:
                        print(f"Created metadata for {idx + 1} samples...")
                print(f"✓ Successfully created metadata for {len(data)} samples")
                return data
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SUN397 dataset from Hugging Face: {str(e)}\n"
                "Please check your internet connection and try again."
            ) from e
    
    def _load_data_tensorflow(self, **kwargs):
        """Fallback: Load SUN397 data using TensorFlow Datasets."""
        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "TensorFlow Datasets is required for SUN397 dataset fallback. "
                "Install it with: pip install tensorflow-datasets"
            ) from e

        # This is the old implementation - kept as fallback
        try:
            ds = tfds.load('sun397', split=self.split, as_supervised=True, 
                          data_dir=self.root_path, download=True)
              # Get dataset info for class names
            builder = tfds.builder('sun397', data_dir=self.root_path)
            info = builder.info
            
            data = []
            for idx, (image_tensor, label_tensor) in enumerate(ds):
                image_np = image_tensor.numpy()
                label_idx = label_tensor.numpy()
                image_pil = Image.fromarray(image_np)
                class_name = info.features['label'].int2str(label_idx)
                
                data.append({
                    'image_path': idx,
                    'label': class_name,
                    '_pil_image': image_pil
                })
            
            return data
            
        except Exception as e:
            raise RuntimeError(
                "Both Hugging Face and TensorFlow Datasets failed to load SUN397. "
                f"Last error: {str(e)}"
            ) from e

    def _get_classes(self) -> List[str]:
        """Get all 397 SUN397 class names from the dataset."""
        try:
            # If we have the HuggingFace dataset loaded, extract real class names
            if hasattr(self, '_hf_dataset') and self._hf_dataset is not None:
                # Use the proper HuggingFace way to get class names
                if hasattr(self._hf_dataset, 'features') and 'label' in self._hf_dataset.features:
                    label_feature = self._hf_dataset.features['label']
                    if hasattr(label_feature, 'names'):
                        # This is the correct way for ClassLabel features
                        class_names = label_feature.names
                        print(f"✓ Extracted {len(class_names)} real class names from SUN397 dataset features")
                        
                        # Clean the class names to make them more readable for CLIP
                        cleaned_names = [self._clean_class_name(name) for name in class_names]
                        return cleaned_names
                
                # Fallback: try to extract from dataset info
                if hasattr(self._hf_dataset, 'info') and hasattr(self._hf_dataset.info, 'features'):
                    label_feature = self._hf_dataset.info.features.get('label')
                    if label_feature and hasattr(label_feature, 'names'):
                        class_names = label_feature.names
                        print(f"✓ Extracted {len(class_names)} real class names from SUN397 dataset info")
                        cleaned_names = [self._clean_class_name(name) for name in class_names]
                        return cleaned_names
                
                print("Warning: Could not find ClassLabel feature with names in HuggingFace dataset")
            
            # Fallback: return placeholder classes
            print("Warning: Could not extract real class names from dataset, using placeholders")
            return [f"class_{i}" for i in range(397)]
            
        except Exception as e:
            print(f"Warning: Error extracting class names: {e}")
            return [f"class_{i}" for i in range(397)]
    
    def _clean_class_name(self, raw_label: str) -> str:
        """Clean and convert raw label to readable class name.
        
        Args:
            raw_label: Raw label like '/a/abbey' or '/a/airplane_cabin'
            
        Returns:
            Cleaned class name like 'abbey' or 'airplane cabin'
        """
        # Remove leading slashes and path components
        cleaned = raw_label.strip('/')
        
        # Split by '/' and take the last part (actual class name)
        parts = cleaned.split('/')
        if len(parts) > 1:
            # Take the last part which should be the class name
            class_name = parts[-1]
        else:
            class_name = cleaned
        
        # Handle special cases and clean up
        class_name = class_name.replace('_', ' ')  # Convert underscores to spaces
        class_name = class_name.replace('-', ' ')  # Convert hyphens to spaces
        
        # Remove common suffixes like '/outdoor', '/indoor'
        suffixes_to_remove = ['/outdoor', '/indoor', 'outdoor', 'indoor']
        for suffix in suffixes_to_remove:
            if class_name.endswith(suffix):
                class_name = class_name[:-len(suffix)].strip()
        
        # Clean up extra spaces
        class_name = ' '.join(class_name.split())
        
        return class_name

    def get_templates(self) -> List[str]:
        """Templates for SUN397 scene classification."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()

    def __getitem__(self, idx: int) -> tuple:
        """Override to handle lazy loading of images from HuggingFace dataset with streaming support."""
        item = self.data[idx]
        
        from PIL import Image
        
        # Check if image is pre-loaded (for backward compatibility)
        if '_pil_image' in item:
            image = item['_pil_image']
            label_text = item['label']
        else:
            # Check if we're in streaming mode and use on-demand loading
            if hasattr(self, '_dataset_for_streaming') and self._dataset_for_streaming is not None:
                # Streaming mode: load image on-demand (即用即扔)
                image, label_text = self._load_image_on_demand(idx)
            else:
                # Non-streaming mode: traditional lazy loading
                image, label_text = self._load_image_traditional(idx)

        if self.transform:
            image = self.transform(image)

        # Convert string label to index
        if label_text in self.class_to_idx:
            label = self.class_to_idx[label_text]
        else:
            # If label not found, use 0 as default
            label = 0
        
        return image, label

    def _load_image_on_demand(self, idx: int) -> tuple:
        """Load image on-demand for streaming mode to save memory (即用即扔)."""
        try:
            if self._dataset_for_streaming is None:
                raise RuntimeError("Streaming dataset not available")
            
            # For streaming datasets, we need to iterate to the specific index
            for i, sample in enumerate(self._dataset_for_streaming):
                if i == idx:
                    image = sample['image']
                    raw_label = sample['label']
                    
                    # Convert image to RGB if needed
                    if hasattr(image, 'convert'):
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                    
                    # Convert label
                    if isinstance(raw_label, int):
                        # Try to get label name from features if available
                        try:
                            if (hasattr(self._dataset_for_streaming, 'features') and 
                                'label' in self._dataset_for_streaming.features and
                                hasattr(self._dataset_for_streaming.features['label'], 'int2str')):
                                label_text = self._dataset_for_streaming.features['label'].int2str(raw_label)
                                label_text = self._clean_class_name(label_text)
                            else:
                                label_text = f"class_{raw_label}"
                        except:
                            label_text = f"class_{raw_label}"
                    elif isinstance(raw_label, str):
                        label_text = self._clean_class_name(raw_label)
                    else:
                        label_text = f"class_{raw_label}"
                    
                    return image, label_text
                elif i > idx:
                    break
            
            # If we can't find the image, return a placeholder
            print(f"Warning: Could not load image at index {idx}, using placeholder")
            from PIL import Image
            return Image.new('RGB', (224, 224), color='gray'), 'unknown'
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            from PIL import Image
            return Image.new('RGB', (224, 224), color='gray'), 'unknown'

    def _load_image_traditional(self, idx: int) -> tuple:
        """Traditional lazy loading for non-streaming mode."""
        item = self.data[idx]
        hf_idx = item['image_path']  # This is the index in the HF dataset
        
        try:
            # Try different methods to access the dataset item
            if hasattr(self._hf_dataset, '__getitem__'):
                hf_item = self._hf_dataset[hf_idx]
            elif hasattr(self._hf_dataset, 'select'):
                hf_item = self._hf_dataset.select([hf_idx])[0]
            else:
                raise AttributeError("Dataset doesn't support indexing")
                
            image = hf_item['image']
            raw_label = hf_item['label']
            
            # Use the proper HuggingFace way to convert label index to name
            if isinstance(raw_label, int):
                # Use ClassLabel.int2str() method to get the real class name
                if (hasattr(self._hf_dataset, 'features') and 
                    'label' in self._hf_dataset.features and
                    hasattr(self._hf_dataset.features['label'], 'int2str')):
                    try:
                        label_text = self._hf_dataset.features['label'].int2str(raw_label)
                        label_text = self._clean_class_name(label_text)
                    except:
                        label_text = f"class_{raw_label}"
                else:
                    label_text = f"class_{raw_label}"
            elif isinstance(raw_label, str):
                # If already a string, clean it
                label_text = self._clean_class_name(raw_label)
            else:
                label_text = f"class_{raw_label}"
                
        except Exception as e:
            # Fallback: create placeholder
            from PIL import Image
            image = Image.new('RGB', (224, 224), color='gray')
            label_text = 'unknown'
            
        return image, label_text
