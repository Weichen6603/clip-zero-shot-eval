"""CIFAR-100 dataset adapter."""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


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
        # Import here to avoid circular import
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()

    def __getitem__(self, idx: int) -> tuple:
        """Override to handle torchvision dataset."""
        item = self.data[idx]
        image, _ = item['_torch_data']

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[item['label']]
        return image, label
