#!/usr/bin/env python3
"""
Test script for ImageNet Hugging Face adapter
"""

import os
import sys

# Add project root to path
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

def test_imagenet_adapter():
    """Test the ImageNet adapter with Hugging Face dataset."""
    print("Testing ImageNet adapter with Hugging Face dataset...")
    
    try:
        from adapters.imagenet_adapter import ImageNetAdapter
        from dataset_adapters import DatasetAdapterRegistry
        
        # Test if adapter is registered
        print(f"Available adapters: {DatasetAdapterRegistry.list_adapters()}")
        
        # Test adapter creation
        print("\nCreating ImageNet adapter...")
        adapter = ImageNetAdapter(
            root_path="/mnt/d/data/imagenet",
            split="validation",
            use_auth_token=True
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(adapter)}")
        print(f"Number of classes: {len(adapter.classes)}")
        print(f"First 10 classes: {adapter.classes[:10]}")
        
        # Test getting a sample
        print("\nTesting sample retrieval...")
        sample_image, sample_label = adapter[0]
        print(f"Sample 0 - Image shape: {sample_image.size if hasattr(sample_image, 'size') else 'No size attr'}")
        print(f"Sample 0 - Label: {sample_label}")
        print(f"Sample 0 - Class name: {adapter.classes[sample_label] if sample_label < len(adapter.classes) else 'Unknown'}")
        
        # Test templates
        templates = adapter.get_templates()
        print(f"\nNumber of templates: {len(templates)}")
        print(f"First 3 templates: {templates[:3]}")
        
        print("\n✅ ImageNet adapter test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ImageNet adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imagenet_adapter()
    sys.exit(0 if success else 1)
