### dataset_factory.py
# Factory class for creating dataset instances
###

from typing import Dict, Any, Type, List
from base_dataset import BaseDatasetAdapter
from dataset_adapters import (
    CIFAR10Adapter,
    ImageNetAdapter,
    CustomDatasetAdapter
)


class DatasetFactory:
    """Factory class for creating dataset adapters.

    This factory manages the registration and creation of dataset adapters,
    allowing easy extension with new dataset types.
    """

    # Registry of available dataset adapters
    _adapters: Dict[str, Type[BaseDatasetAdapter]] = {
        'cifar10': CIFAR10Adapter,
        'imagenet': ImageNetAdapter,
        'custom': CustomDatasetAdapter,
    }

    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseDatasetAdapter]):
        """Register a new dataset adapter.

        Args:
            name: Name to register the adapter under
            adapter_class: Adapter class (must inherit from BaseDatasetAdapter)
        """
        if not issubclass(adapter_class, BaseDatasetAdapter):
            raise ValueError(f"{adapter_class} must inherit from BaseDatasetAdapter")

        cls._adapters[name] = adapter_class

    @classmethod
    def list_adapters(cls) -> List[str]:
        """Get list of registered adapter names."""
        return list(cls._adapters.keys())

    @classmethod
    def create_dataset(cls, config: Dict[str, Any], transform=None) -> BaseDatasetAdapter:
        """Create a dataset adapter from configuration.

        Args:
            config: Dataset configuration dictionary containing at least:
                - 'type': str, type of dataset adapter to use
                - 'root_path': str, root path of the dataset
                - Additional adapter-specific parameters
            transform: Image transformations to apply

        Returns:
            Initialized dataset adapter
        """
        # Make a copy to avoid modifying the original config
        config = config.copy()

        # Extract dataset type
        dataset_type = config.pop('type')
        if dataset_type not in cls._adapters:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Available types: {cls.list_adapters()}"
            )

        # Get adapter class
        adapter_class = cls._adapters[dataset_type]

        # Create and return adapter instance
        return adapter_class(transform=transform, **config)