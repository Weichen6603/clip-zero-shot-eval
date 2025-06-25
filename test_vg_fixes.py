#!/usr/bin/env python3
"""
Quick test for Visual Genome lazy loading fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adapters.visual_genome_adapter import VisualGenomeAdapter

def test_sample_discovery():
    """Test if the adapter can find the expected number of samples."""
    print("🧪 Testing Visual Genome Sample Discovery")
    print("=" * 50)
    
    dataset1_size = 0
    dataset1_classes = 0
    
    # Test with limited samples
    print("\n📊 Test 1: Limited samples (max_samples=1000)")
    config1 = {
        'root_path': '/mnt/d/data/visual_genome',
        'max_samples': 1000,
        'min_objects': 1,
        'max_objects': 20,
        'use_synsets': False,
        'config_name': 'objects_v1.2.0'
    }
    
    try:
        dataset1 = VisualGenomeAdapter(**config1)
        dataset1_size = len(dataset1)
        dataset1_classes = len(dataset1.classes)
        print(f"✅ Found {dataset1_size} samples (target: ~1000)")
        print(f"✅ Discovered {dataset1_classes} classes")
        
        if len(dataset1) > 0:
            print(f"✅ Sample classes: {dataset1.classes[:10]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with no sample limit  
    print(f"\n📊 Test 2: No sample limit (should find more samples)")
    config2 = {
        'root_path': '/mnt/d/data/visual_genome',
        # No max_samples - should scan more comprehensively
        'min_objects': 1,
        'max_objects': 20,
        'use_synsets': False,
        'config_name': 'objects_v1.2.0'
    }
    
    try:
        dataset2 = VisualGenomeAdapter(**config2)
        print(f"✅ Found {len(dataset2)} samples (should be > 1000)")
        print(f"✅ Discovered {len(dataset2.classes)} classes")
        
        if len(dataset2) > dataset1_size:
            print(f"✅ Success: Found more samples without limit ({len(dataset2)} vs {dataset1_size})")
        else:
            print(f"⚠️  Warning: Sample count not increased as expected")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sample_discovery()
