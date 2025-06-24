#!/usr/bin/env python3
"""
Test script for official ImageNet-1K adapter
"""

import os
import sys

# Add project root to path
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

def test_official_imagenet():
    """Test the ImageNet adapter with official imagenet-1k dataset."""
    print("Testing ImageNet adapter with official imagenet-1k dataset...")
    
    try:
        from adapters.imagenet_adapter import ImageNetAdapter
        from dataset_adapters import DatasetAdapterRegistry
        
        # Test if adapter is registered
        print(f"Available adapters: {DatasetAdapterRegistry.list_adapters()}")
        
        # Test adapter creation 
        print("\nCreating ImageNet adapter...")
        print("Note: This will download the official ImageNet-1K dataset which requires authentication.")
        print("Make sure you have:")
        print("1. Logged in with: huggingface-cli login")
        print("2. Requested access to the imagenet-1k dataset")
        
        adapter = ImageNetAdapter(
            root_path="/mnt/d/data/imagenet",
            split="validation"
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(adapter)}")
        print(f"Number of classes: {len(adapter.classes)}")
        print(f"First 5 classes: {adapter.classes[:5]}")
        
        # Test getting sample data
        print("\nTesting sample data...")
        first_item = adapter.data[0]
        print(f"Sample 0 - Label: {first_item['label']}")
        print(f"Sample 0 - Synset: {first_item['synset']}")
        print(f"Sample 0 - Image path: {first_item['image_path']}")
        
        # Test templates
        templates = adapter.get_templates()
        print(f"\nNumber of templates: {len(templates)}")
        print(f"First 3 templates: {templates[:3]}")
        
        print("\n✅ Official ImageNet adapter test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ImageNet adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_official_imagenet()
    sys.exit(0 if success else 1)
