### base_dataset.py
# Concrete dataset adapter implementations
###

import os
import json
import csv
from typing import List, Dict, Any
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