"""
Dataset adapters package.

This package contains all dataset adapter implementations.
Each adapter should inherit from BaseDatasetAdapter and implement
the required abstract methods.
"""

# Make adapters importable from the package level
from .cifar10_adapter import CIFAR10Adapter
from .cifar100_adapter import CIFAR100Adapter
from .imagenet_adapter import ImageNetAdapter
from .sun397_adapter import SUN397Adapter

__all__ = [
    'CIFAR10Adapter',
    'CIFAR100Adapter', 
    'ImageNetAdapter',
    'SUN397Adapter'
]
