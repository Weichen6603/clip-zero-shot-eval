#!/usr/bin/env python3
"""
Simple test script for ImageNet Hugging Face adapter
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
        
        # Test adapter creation with smaller subset first
        print("\nCreating ImageNet adapter...")
        adapter = ImageNetAdapter(
            root_path="/mnt/d/data/imagenet",
            split="validation"
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(adapter)}")
        print(f"Number of classes: {len(adapter.classes)}")
        print(f"First 5 classes: {adapter.classes[:5]}")
        
        # Test getting a sample
        print("\nTesting sample retrieval...")
        first_item = adapter.data[0]
        print(f"Sample 0 - Label: {first_item['label']}")
        print(f"Sample 0 - Label text: {first_item['label_text']}")
        print(f"Sample 0 - Image type: {type(first_item['image_obj'])}")
        
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
