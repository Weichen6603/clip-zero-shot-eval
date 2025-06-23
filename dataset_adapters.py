### dataset_adapters.py
# Concrete dataset adapter implementations
###

import os
import json
import csv
from typing import List
from base_dataset import BaseDatasetAdapter

class CIFAR10Adapter(BaseDatasetAdapter):
    """Adapter for CIFAR-10 dataset."""

    def _load_data(self, **kwargs):
        """Load CIFAR-10 data using torchvision."""
        from torchvision.datasets import CIFAR10

        # Use torchvision's CIFAR10 dataset
        dataset = CIFAR10(root=self.root_path, train=(self.split=='train'), download=True)

        data = []
        for idx in range(len(dataset)):
            # Create a temporary path for compatibility
            data.append({
                'image_path': idx,  # We'll handle this in __getitem__
                'label': dataset.classes[dataset[idx][1]],
                '_torch_data': dataset[idx]  # Store original data
            })

        self._torch_dataset = dataset
        return data

    def _get_classes(self) -> List[str]:
        """CIFAR-10 class names."""
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    def get_templates(self) -> List[str]:
        """Templates for CIFAR-10 zero-shot classification."""
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a black and white photo of a {}.",
            "a low contrast photo of a {}.",
            "a high contrast photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of a small {}.",
            "a photo of a big {}.",
            "a photo of the {}.",
            "a cropped photo of a {}.",
            "a bright photo of a {}.",
            "a dark photo of a {}.",
            "a photo of my {}.",
            "a photo of the cool {}.",
            "a close-up photo of a {}.",
            "a painting of a {}.",
            "a rendering of a {}.",
            "a drawing of a {}.",
            "a tattoo of a {}.",
        ]

    def __getitem__(self, idx: int) -> tuple:
        """Override to handle torchvision dataset."""
        item = self.data[idx]
        image, _ = item['_torch_data']

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[item['label']]
        return image, label


class CIFAR100Adapter(BaseDatasetAdapter):
    """Adapter for CIFAR-100 dataset."""

    def _load_data(self, use_coarse_labels: bool = False, **kwargs):
        """Load CIFAR-100 data using torchvision.
        
        Args:
            use_coarse_labels: If True, use 20 coarse labels instead of 100 fine labels
        """
        from torchvision.datasets import CIFAR100

        # Use torchvision's CIFAR100 dataset
        dataset = CIFAR100(root=self.root_path, train=(self.split=='train'), download=True)
        
        self.use_coarse_labels = use_coarse_labels
        
        # CIFAR-100 coarse label mapping (fine label -> coarse label)
        coarse_label_mapping = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ]
        
        data = []
        for idx in range(len(dataset)):
            # Get fine label
            _, fine_label = dataset[idx]
            coarse_label = coarse_label_mapping[fine_label]
            
            # Choose which label to use
            if use_coarse_labels:
                label_name = self._get_coarse_classes()[coarse_label]
            else:
                label_name = self._get_fine_classes()[fine_label]
            
            data.append({
                'image_path': idx,  # We'll handle this in __getitem__
                'label': label_name,
                '_torch_data': dataset[idx],  # Store original data
                'fine_label': fine_label,
                'coarse_label': coarse_label
            })

        self._torch_dataset = dataset
        return data

    def _get_classes(self) -> List[str]:
        """Get CIFAR-100 class names based on label type."""
        if hasattr(self, 'use_coarse_labels') and self.use_coarse_labels:
            return self._get_coarse_classes()
        else:
            return self._get_fine_classes()
    
    def _get_fine_classes(self) -> List[str]:
        """CIFAR-100 fine class names (100 classes)."""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
    
    def _get_coarse_classes(self) -> List[str]:
        """CIFAR-100 coarse class names (20 superclasses)."""
        return [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
            'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
            'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]

    def get_templates(self) -> List[str]:
        """Templates for CIFAR-100 zero-shot classification."""
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a black and white photo of a {}.",
            "a low contrast photo of a {}.",
            "a high contrast photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of a small {}.",
            "a photo of a big {}.",
            "a photo of the {}.",
            "a cropped photo of a {}.",
            "a bright photo of a {}.",
            "a dark photo of a {}.",
            "a photo of my {}.",
            "a photo of the cool {}.",
            "a close-up photo of a {}.",
            "a painting of a {}.",
            "a rendering of a {}.",
            "a drawing of a {}.",
            "a tattoo of a {}.",
            "a sketch of a {}.",
            "a doodle of a {}.",
            "a cartoon {}.",
            "art of a {}.",
            "graffiti of a {}.",
        ]

    def __getitem__(self, idx: int) -> tuple:
        """Override to handle torchvision dataset."""
        item = self.data[idx]
        image, _ = item['_torch_data']

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[item['label']]
        return image, label


class ImageNetAdapter(BaseDatasetAdapter):
    """Adapter for ImageNet dataset."""

    def _load_data(self, class_map_file: str = None, **kwargs):
        """Load ImageNet data.

        Args:
            class_map_file: Path to JSON file mapping folder names to class names
        """
        # Load class mapping if provided
        if class_map_file and os.path.exists(class_map_file):
            with open(class_map_file, 'r') as f:
                self.folder_to_class = json.load(f)
        else:
            # Use folder names as class names
            self.folder_to_class = {}

        data = []
        image_dir = os.path.join(self.root_path, self.split)

        # Iterate through class folders
        for class_folder in sorted(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Get class name
            class_name = self.folder_to_class.get(class_folder, class_folder)

            # Add all images in the folder
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'image_path': os.path.join(class_path, img_name),
                        'label': class_name,
                        'folder': class_folder
                    })

        return data

    def _get_classes(self) -> List[str]:
        """Get ImageNet class names."""
        # Extract unique class names from data
        classes = sorted(list(set(item['label'] for item in self.data)))
        return classes

    def get_templates(self) -> List[str]:
        """Templates for ImageNet zero-shot classification."""
        return [
            "a bad photo of a {}.",
            "a photo of many {}.",
            "a sculpture of a {}.",
            "a photo of the hard to see {}.",
            "a low resolution photo of the {}.",
            "a rendering of a {}.",
            "graffiti of a {}.",
            "a bad photo of the {}.",
            "a cropped photo of the {}.",
            "a tattoo of a {}.",
            "the embroidered {}.",
            "a photo of a hard to see {}.",
            "a bright photo of a {}.",
            "a photo of a clean {}.",
            "a photo of a dirty {}.",
            "a dark photo of the {}.",
            "a drawing of a {}.",
            "a photo of my {}.",
            "the plastic {}.",
            "a photo of the cool {}.",
            "a close-up photo of a {}.",
            "a black and white photo of the {}.",
            "a painting of the {}.",
            "a painting of a {}.",
            "a pixelated photo of the {}.",
            "a sculpture of the {}.",
            "a bright photo of the {}.",
            "a cropped photo of a {}.",
            "a plastic {}.",
            "a photo of the dirty {}.",
            "a jpeg corrupted photo of a {}.",
            "a blurry photo of the {}.",
            "a photo of the {}.",
            "a good photo of the {}.",
            "a rendering of the {}.",
            "a {} in a video game.",
            "a photo of one {}.",
            "a doodle of a {}.",
            "a close-up photo of the {}.",
            "a photo of a {}.",
            "the origami {}.",
            "the {} in a video game.",
            "a sketch of a {}.",
            "a doodle of the {}.",
            "a origami {}.",
            "a low resolution photo of a {}.",
            "the toy {}.",
            "a rendition of the {}.",
            "a photo of the clean {}.",
            "a photo of a large {}.",
            "a rendition of a {}.",
            "a photo of a nice {}.",
            "a photo of a weird {}.",
            "a blurry photo of a {}.",
            "a cartoon {}.",
            "art of a {}.",
            "a sketch of the {}.",
            "a embroidered {}.",
            "a pixelated photo of a {}.",
            "itap of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a good photo of a {}.",
            "a plushie {}.",
            "a photo of the nice {}.",
            "a photo of the small {}.",
            "a photo of the weird {}.",
            "the cartoon {}.",
            "art of the {}.",
            "a drawing of the {}.",
            "a photo of the large {}.",
            "a black and white photo of a {}.",
            "the plushie {}.",
            "a dark photo of a {}.",
            "itap of a {}.",
            "graffiti of the {}.",
            "a toy {}.",
            "itap of my {}.",
            "a photo of a cool {}.",
            "a photo of a small {}.",
            "a tattoo of the {}.",
        ]


class CustomDatasetAdapter(BaseDatasetAdapter):
    """Adapter for custom datasets with flexible format."""

    def _load_data(self,
                   annotation_file: str,
                   annotation_format: str = 'json',
                   image_dir: str = None,
                   image_key: str = 'image',
                   label_key: str = 'label',
                   **kwargs):
        """Load custom dataset from various formats.

        Args:
            annotation_file: Path to annotation file
            annotation_format: Format of annotation file (json/csv/txt)
            image_dir: Directory containing images (if paths in annotations are relative)
            image_key: Key/column name for image paths
            label_key: Key/column name for labels
        """
        data = []
        image_dir = image_dir or self.root_path

        if annotation_format == 'json':
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            for item in annotations:
                image_path = item[image_key]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(image_dir, image_path)

                data.append({
                    'image_path': image_path,
                    'label': item[label_key]
                })

        elif annotation_format == 'csv':
            with open(annotation_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_path = row[image_key]
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(image_dir, image_path)

                    data.append({
                        'image_path': image_path,
                        'label': row[label_key]
                    })

        elif annotation_format == 'txt':
            # Assume format: image_path label (space-separated)
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = ' '.join(parts[1:])  # Handle multi-word labels

                        if not os.path.isabs(image_path):
                            image_path = os.path.join(image_dir, image_path)

                        data.append({
                            'image_path': image_path,
                            'label': label
                        })

        return data

    def _get_classes(self) -> List[str]:
        """Extract unique classes from data."""
        classes = sorted(list(set(item['label'] for item in self.data)))
        return classes

    def get_templates(self) -> List[str]:
        """Default templates for custom datasets."""
        return [
            "a photo of a {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "a photograph of a {}.",
            "{} in a photo.",
            "a photo showing a {}.",
            "a visual representation of a {}.",
            "this is a {}.",
            "a clear photo of a {}.",
            "a good example of a {}.",
        ]


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
    
    def _load_data_huggingface(self, test_split_ratio=0.2, **kwargs):
        """Load SUN397 data using Hugging Face Datasets.
        
        Args:
            test_split_ratio: Ratio to split train data into train/test (default: 0.2)
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
                # Load full train dataset and split it
                print("Note: SUN397 only has 'train' split. Creating test split from training data...")
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path)
                
                # Split the dataset
                train_test_split = dataset.train_test_split(test_size=test_split_ratio, shuffle=True, seed=42)
                dataset = train_test_split['test']  # Use test portion
                print(f"Using test split ({len(dataset)} samples) from training data")
                
            elif self.split == 'train':
                # Load full train dataset
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path)
                
                # Optionally split to keep only training portion
                if test_split_ratio > 0:
                    train_test_split = dataset.train_test_split(test_size=test_split_ratio, shuffle=True, seed=42)
                    dataset = train_test_split['train']  # Use train portion
                    print(f"Using train split ({len(dataset)} samples) from training data")
                else:
                    print(f"Using full dataset ({len(dataset)} samples)")
            else:
                # Fallback: use available split
                print(f"Warning: Split '{self.split}' not available. Using 'train' split.")
                dataset = load_dataset("1aurent/SUN397", split='train', cache_dir=self.root_path)
              # Apply max_samples limit if specified
            max_samples = kwargs.get('max_samples', None)
            dataset_length = len(dataset) if hasattr(dataset, '__len__') else 87003
            
            if max_samples and max_samples < dataset_length:
                print(f"Limited to {max_samples} samples for testing")
                actual_length = max_samples
            else:
                print(f"Using full dataset ({dataset_length} samples)")
                actual_length = dataset_length
            
            # Store the dataset reference for lazy loading
            self._hf_dataset = dataset
            
            # Create lightweight data structure WITHOUT loading actual images
            data = []
            print(f"Creating metadata for {actual_length} samples...")
            
            # Only create metadata entries - don't load images yet
            for idx in range(actual_length):
                # We'll get the label when needed in __getitem__
                data.append({
                    'image_path': idx,  # Index into the HF dataset
                    'label': None,  # Will be loaded on demand
                })
                
                # Show progress for large datasets
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
        return [
            "a photo of a {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "a photograph of a {}.",
            "a scene of a {}.",
            "a view of a {}.",
            "this is a {}.",
            "this shows a {}.",
            "a photo taken in a {}.",
            "a photo taken at a {}.",
            "an outdoor photo of a {}.",            "an indoor photo of a {}.",
            "a scenic view of a {}.",
            "a landscape photo of a {}.",
            "architecture photo of a {}.",            "a wide shot of a {}.",
            "a close-up of a {}.",
            "a detailed view of a {}.",
            "a beautiful photo of a {}.",
            "a typical {}.",
        ]

    def __getitem__(self, idx: int) -> tuple:
        """Override to handle lazy loading of images from HuggingFace dataset."""
        item = self.data[idx]
        
        from PIL import Image        # Check if image is pre-loaded (for backward compatibility)
        if '_pil_image' in item:
            image = item['_pil_image']
            label_text = item['label']
        else:
            # Lazy load from HuggingFace dataset
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
                image = Image.new('RGB', (224, 224), color='gray')
                label_text = 'unknown'

        if self.transform:
            image = self.transform(image)

        # Convert string label to index
        if label_text in self.class_to_idx:
            label = self.class_to_idx[label_text]
        else:
            # If label not found, use 0 as default
            label = 0
        
        return image, label