#!/usr/bin/env python3
"""
Extended inspect of ImageNet dataset to find labels
"""

import os
import sys

# Add project root to path
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

def inspect_imagenet_detailed():
    """Detailed inspection of the ImageNet dataset."""
    print("Detailed inspection of ImageNet dataset...")
    
    try:
        from datasets import load_dataset
        import os
        
        print("Loading dataset...")
        dataset = load_dataset("mlx-vision/imagenet-1k", split="validation", cache_dir="/mnt/d/data/imagenet")
        
        print(f"Dataset info: {dataset}")
        
        # Check if there's any metadata or filename information
        print("\nChecking for additional metadata...")
        
        # Look at the dataset builder or config
        if hasattr(dataset, 'info') and dataset.info:
            print(f"Dataset info object: {dataset.info}")
            if hasattr(dataset.info, 'features'):
                print(f"Info features: {dataset.info.features}")
        
        # Check the cache directory for any label files
        cache_dir = "/mnt/d/data/imagenet"
        if os.path.exists(cache_dir):
            print(f"\nCache directory contents:")
            for root, dirs, files in os.walk(cache_dir):
                level = root.replace(cache_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Show first 10 files
                    print(f"{subindent}{file}")
                if len(files) > 10:
                    print(f"{subindent}... and {len(files) - 10} more files")
        
        # Try a different approach - check the original ImageNet structure
        print("\nTrying standard ImageNet validation dataset approach...")
        try:
            # Try loading with standard ImageNet structure
            from torchvision.datasets import ImageNet
            print("Checking if torchvision ImageNet is available...")
            # This would need the full ImageNet dataset structure
        except Exception as e:
            print(f"Torchvision ImageNet not available: {e}")
        
        # Check if there are any cached label files
        label_files = [f for f in os.listdir(cache_dir) if 'label' in f.lower() or 'class' in f.lower()]
        if label_files:
            print(f"Found potential label files: {label_files}")
        
        # Look for any JSON or text files that might contain class mapping
        meta_files = []
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(('.json', '.txt', '.csv')):
                    meta_files.append(os.path.join(root, file))
        
        if meta_files:
            print(f"\nFound metadata files: {meta_files[:5]}")  # Show first 5
            
        return True
        
    except Exception as e:
        print(f"Error in detailed inspection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = inspect_imagenet_detailed()
    print("\n" + "=" * 50)
    if success:
        print("✓ Detailed inspection completed.")
    else:
        print("✗ Detailed inspection failed.")
