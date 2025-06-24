"""CIFAR-10 dataset adapter."""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
