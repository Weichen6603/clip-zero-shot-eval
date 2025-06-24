### dataset_factory.py
# Factory class for creating dataset instances
###

from typing import Dict, Any, Type, List
from base_dataset import BaseDatasetAdapter
from dataset_adapters import DatasetAdapterRegistry


class DatasetFactory:
    """Factory class for creating dataset adapters.

    This factory manages the registration and creation of dataset adapters,
    allowing easy extension with new dataset types.
    """

    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseDatasetAdapter]):
        """Register a new dataset adapter.

        Args:
            name: Name to register the adapter under
            adapter_class: Adapter class (must inherit from BaseDatasetAdapter)
        """
        DatasetAdapterRegistry.register(name, adapter_class)

    @classmethod
    def list_adapters(cls) -> List[str]:
        """Get list of registered adapter names."""
        return DatasetAdapterRegistry.list_adapters()

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
            Initialized dataset adapter        """
        # Make a copy to avoid modifying the original config
        import copy
        if hasattr(config, '__dict__'):
            # If config is a dataclass object, get its dict representation
            config_dict = copy.deepcopy(config.__dict__)
        else:
            # If config is already a dict, use it directly
            config_dict = copy.deepcopy(config)        # Extract dataset type
        dataset_type = config_dict.pop('type')
        
        # Get adapter class from registry
        adapter_class = DatasetAdapterRegistry.get_adapter(dataset_type)

        # Create and return adapter instance
        return adapter_class(transform=transform, **config_dict)