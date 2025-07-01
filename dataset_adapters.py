### dataset_adapters.py
# Central registry for dataset adapters and shared templates
###

from typing import List, Dict, Type
from base_dataset import BaseDatasetAdapter

class DatasetTemplates:
    """Centralized templates for different types of datasets."""
    
    @staticmethod
    def get_templates() -> List[str]:
        """Unified templates for all datasets - comprehensive set optimized for zero-shot classification."""
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
            "a photo of many {}.",
            "a sculpture of a {}.",
            "a photo of the hard to see {}.",
            "a low resolution photo of the {}.",
            "a bad photo of the {}.",
            "a cropped photo of the {}.",
            "the embroidered {}.",
            "a photo of a hard to see {}.",
            "a photo of a clean {}.",
            "a photo of a dirty {}.",
            "a dark photo of the {}.",
            "the plastic {}.",
            "a black and white photo of the {}.",
            "a painting of the {}.",
            "a pixelated photo of the {}.",
            "a sculpture of the {}.",
            "a bright photo of the {}.",
            "a plastic {}.",
            "a photo of the dirty {}.",
            "a jpeg corrupted photo of a {}.",
            "a blurry photo of the {}.",
            "a good photo of the {}.",
            "a rendering of the {}.",
            "a {} in a video game.",
            "a photo of one {}.",
            "a doodle of a {}.",
            "a close-up photo of the {}.",
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
            "art of a {}.",
            "a sketch of the {}.",
            "a embroidered {}.",
            "a pixelated photo of a {}.",
            "itap of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a plushie {}.",
            "a photo of the nice {}.",
            "a photo of the small {}.",
            "a photo of the weird {}.",
            "the cartoon {}.",
            "art of the {}.",
            "a drawing of the {}.",
            "a photo of the large {}.",
            "the plushie {}.",
            "itap of a {}.",
            "a toy {}.",
            "itap of my {}.",
            "a photo of a cool {}.",
            "a tattoo of the {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "a photograph of a {}.",
            "a scene of a {}.",
            "a view of a {}.",
            "this is a {}.",
            "this shows a {}.",
            "a photo taken in a {}.",
            "a photo taken at a {}.",
            "an outdoor photo of a {}.",
            "an indoor photo of a {}.",
            "a scenic view of a {}.",
            "a landscape photo of a {}.",
            "architecture photo of a {}.",
            "a wide shot of a {}.",
            "a detailed view of a {}.",
            "a beautiful photo of a {}.",
            "a typical {}.",
            "{} in a photo.",
            "a photo showing a {}.",
            "a visual representation of a {}.",
            "a clear photo of a {}.",
            "a good example of a {}.",
        ]
    



class DatasetAdapterRegistry:
    """Registry for dataset adapters."""
    
    _adapters: Dict[str, Type[BaseDatasetAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[BaseDatasetAdapter]):
        """Register a dataset adapter."""
        cls._adapters[name] = adapter_class
    
    @classmethod
    def get_adapter(cls, name: str) -> Type[BaseDatasetAdapter]:
        """Get a registered adapter by name."""
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' not found. Available adapters: {list(cls._adapters.keys())}")
        return cls._adapters[name]
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapter names."""
        return list(cls._adapters.keys())


# Import and register all adapters
def _register_adapters():
    """Import and register all available adapters."""
    try:
        from adapters.cifar10_adapter import CIFAR10Adapter
        DatasetAdapterRegistry.register('cifar10', CIFAR10Adapter)
    except ImportError:
        pass
    
    try:
        from adapters.cifar100_adapter import CIFAR100Adapter
        DatasetAdapterRegistry.register('cifar100', CIFAR100Adapter)
    except ImportError:
        pass
    
    try:
        from adapters.imagenet_adapter import ImageNetAdapter
        DatasetAdapterRegistry.register('imagenet', ImageNetAdapter)
    except ImportError:
        pass
    
    try:
        from adapters.sun397_adapter import SUN397Adapter
        DatasetAdapterRegistry.register('sun397', SUN397Adapter)
    except ImportError:
        pass
    
    try:
        from adapters.visual_genome_adapter import VisualGenomeAdapter
        DatasetAdapterRegistry.register('visual_genome', VisualGenomeAdapter)
    except ImportError:
        pass
    
    try:
        from adapters.treeoflife_adapter import TreeOfLifeAdapter
        DatasetAdapterRegistry.register('treeoflife', TreeOfLifeAdapter)
    except ImportError:
        pass


# Register adapters on import
_register_adapters()


# Convenience functions for backward compatibility
def get_adapter(name: str) -> Type[BaseDatasetAdapter]:
    """Get a dataset adapter by name."""
    return DatasetAdapterRegistry.get_adapter(name)

def list_adapters() -> List[str]:
    """List all available adapters."""
    return DatasetAdapterRegistry.list_adapters()
