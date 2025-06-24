#!/usr/bin/env python3
"""
Inspect ImageNet dataset structure
"""

import os
import sys

def inspect_dataset():
    """Inspect the actual structure of the ImageNet dataset."""
    print("Inspecting ImageNet dataset structure...")
    
    try:
        from datasets import load_dataset
        
        # Load a small sample to inspect
        print("Loading ImageNet dataset...")
        dataset = load_dataset(
            "mlx-vision/imagenet-1k", 
            split="validation[:10]",  # Load only first 10 samples
            cache_dir="/mnt/d/data/imagenet"
        )
        
        print(f"Dataset features: {dataset.features}")
        print(f"Number of samples: {len(dataset)}")
        
        # Inspect first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"  {key}: {type(value)} - {value if key != 'image' else 'PIL Image object'}")
        
        # Try different splits to see what's available
        print("\nChecking available splits...")
        try:
            info = load_dataset("mlx-vision/imagenet-1k", cache_dir="/mnt/d/data/imagenet")
            print(f"Available splits: {list(info.keys())}")
        except Exception as e:
            print(f"Could not get split info: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    inspect_dataset()
