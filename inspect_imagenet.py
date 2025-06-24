#!/usr/bin/env python3
"""
Inspect ImageNet dataset structure
"""

import os
import sys

# Add project root to path
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

def inspect_imagenet_dataset():
    """Inspect the ImageNet dataset structure."""
    print("Inspecting ImageNet dataset structure...")
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset...")
        dataset = load_dataset("mlx-vision/imagenet-1k", split="validation", cache_dir="/mnt/d/data/imagenet")
        
        print(f"Dataset info: {dataset}")
        print(f"Dataset features: {dataset.features}")
        print(f"Dataset length: {len(dataset)}")
        
        # Check a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Sample structure: {type(sample)}")
            for key, value in sample.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'size'):
                    print(f"    size: {value.size}")
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"    length: {len(value)}")
                else:
                    print(f"    value: {value}")
            
            # Check if there are labels in the image file names or metadata
            print(f"\nFirst few samples:")
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: {sample}")
        
        return True
        
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = inspect_imagenet_dataset()
    print("\n" + "=" * 50)
    if success:
        print("✓ Dataset inspection completed.")
    else:
        print("✗ Dataset inspection failed.")
