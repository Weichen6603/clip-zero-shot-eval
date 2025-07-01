### base_dataset.py
# Base dataset adapter for unified interface
###

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Dict, Tuple, Any, List

class BaseDatasetAdapter(ABC, Dataset):
    """Base class for all dataset adapters.
    
    All dataset adapters should inherit from this class and implement
    the abstract methods to ensure a unified interface for CLIP evaluation.
    """
    
    def __init__(self, root_path: str, transform=None, split: str = 'test', **kwargs):
        """
        Args:
            root_path: Root directory of the dataset
            transform: Image transformations to apply
            split: Dataset split to use (train/val/test)
            **kwargs: Additional dataset-specific arguments
        """
        self.root_path = root_path
        self.transform = transform
        self.split = split
        self.data = self._load_data(**kwargs)
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    @abstractmethod
    def _load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset annotations and file paths.
        
        Returns:
            List of dictionaries containing at least:
            - 'image_path': str, path to image file
            - 'label': str or int, class label
        """
        pass
    
    @abstractmethod
    def _get_classes(self) -> List[str]:
        """Get list of class names in the dataset.
        
        Returns:
            List of class names as strings
        """
        pass
    
    @abstractmethod
    def get_templates(self) -> List[str]:
        """Get prompt templates for zero-shot classification.
        
        Returns:
            List of prompt templates with {} placeholder for class name
            e.g., ["a photo of a {}.", "a picture of {}."]
        """
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Returns:
            image: Transformed image tensor
            label: Integer class label
        """
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index if it's a string
        if isinstance(item['label'], str):
            label = self.class_to_idx[item['label']]
        else:
            label = item['label']
        
        return image, label
    
    def __len__(self) -> int:
        return len(self.data)