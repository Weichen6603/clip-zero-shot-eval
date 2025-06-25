#!/usr/bin/env python3
"""
Test script for Visual Genome lazy loading adapter.
Demonstrates memory-efficient loading of large datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adapters.visual_genome_adapter import VisualGenomeAdapter
import torch
import time
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_lazy_loading():
    """Test the lazy loading Visual Genome adapter."""
    print("🧪 Testing Visual Genome Lazy Loading Adapter")
    print("=" * 60)
    
    # Memory before loading
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test configuration
    config = {
        'root_path': './data/vg_test',  # Local cache directory
        'max_samples': 100,  # Small test size
        'min_objects': 1,
        'max_objects': 10,
        'use_synsets': False,
        'config_name': 'objects_v1.2.0'
    }
    
    print(f"\nLoading Visual Genome with lazy loading...")
    print(f"Configuration: {config}")
    
    start_time = time.time()
    
    try:
        # Create adapter with lazy loading
        dataset = VisualGenomeAdapter(**config)
        
        load_time = time.time() - start_time
        after_load_memory = get_memory_usage()
        after_samples_memory = after_load_memory  # Initialize for scope
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"Load time: {load_time:.2f} seconds")
        print(f"Memory after loading: {after_load_memory:.1f} MB")
        print(f"Memory increase: {after_load_memory - initial_memory:.1f} MB")
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Number of classes: {len(dataset.classes)}")
        
        # Test sample access
        if len(dataset) > 0:
            print(f"\n🔍 Testing sample access:")
            
            # Test first sample
            print("Loading first sample...")
            sample_start = time.time()
            image, label = dataset[0]
            sample_time = time.time() - sample_start
            
            print(f"Sample loaded in {sample_time:.3f} seconds")
            print(f"Image shape: {image.size if hasattr(image, 'size') else 'N/A'}")
            print(f"Label: {label}")
            
            # Test sample info
            sample_info = dataset.get_sample_info(0)
            print(f"Sample info: {sample_info}")
            
            # Test multiple samples to check caching
            print(f"\n⚡ Testing caching with multiple samples:")
            cache_test_start = time.time()
            for i in range(min(5, len(dataset))):
                _, _ = dataset[i]
            cache_test_time = time.time() - cache_test_start
            print(f"Loaded 5 samples in {cache_test_time:.3f} seconds")
            
            # Memory after sample loading
            after_samples_memory = get_memory_usage()
            print(f"Memory after loading samples: {after_samples_memory:.1f} MB")
            
        print(f"\n📊 Memory Efficiency Summary:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  After dataset load: {after_load_memory:.1f} MB (+{after_load_memory - initial_memory:.1f} MB)")
        if len(dataset) > 0:
            print(f"  After sample loading: {after_samples_memory:.1f} MB (+{after_samples_memory - after_load_memory:.1f} MB)")
        print(f"  Total memory increase: {get_memory_usage() - initial_memory:.1f} MB")
        
        # Show sample classes
        if len(dataset.classes) > 0:
            print(f"\n🏷️ Sample classes: {dataset.classes[:10]}...")
        
        print(f"\n✅ Lazy loading test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("Visual Genome Lazy Loading Test")
    print("This test demonstrates memory-efficient loading of large datasets.")
    print("Note: Requires internet connection for first-time dataset download.\n")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    test_lazy_loading()

if __name__ == "__main__":
    main()
