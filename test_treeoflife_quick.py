#!/usr/bin/env python3
"""Quick test for TreeOfLife adapter with grouped classification."""

import sys
import time
from adapters.treeoflife_adapter import TreeOfLifeAdapter

def test_treeoflife_quick():
    """Test TreeOfLife adapter with grouped classes."""
    print("🧪 Quick TreeOfLife adapter test (grouped classification)")
    print("=" * 50)
    
    try:
        # Test with minimal parameters
        print("Creating adapter with max_samples=100...")
        
        adapter = TreeOfLifeAdapter(
            root_path='/mnt/d/data/treeoflife',
            max_samples=100,
            max_shards=3,  # Use 3 shards for more diversity
            taxonomic_level='species',  # Use species level for grouped classes
            min_images_per_class=1,
            exclude_partial_labels=False  # Don't be too strict
        )
        
        print(f"✅ Adapter created successfully!")
        print(f"📊 Dataset size: {len(adapter)} samples")
        print(f"📊 Number of classes: {len(adapter.classes)}")
        print(f"📋 Classes: {adapter.classes}")
        
        # Test getting a sample
        if len(adapter) > 0:
            print(f"\n🔍 Testing sample access...")
            sample_image, sample_label = adapter[0]
            print(f"✅ Sample loaded: image shape {sample_image.shape}, label {sample_label}")
            print(f"   Class name: {adapter.classes[sample_label]}")
        
        # Test templates
        templates = adapter.get_templates()
        print(f"\n📝 Templates: {len(templates)} available")
        print(f"   First template: '{templates[0]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set a timeout for the entire test
    start_time = time.time()
    success = test_treeoflife_quick()
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Test {'PASSED' if success else 'FAILED'} in {elapsed:.1f} seconds")
