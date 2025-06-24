"""ImageNet dataset adapter."""

import os
import sys
import json
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class ImageNetAdapter(BaseDatasetAdapter):
    """Adapter for ImageNet dataset."""

    def _load_data(self, class_map_file: Optional[str] = None, **kwargs):
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
        # Import here to avoid circular import
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()
