#!/usr/bin/env python3
"""
Test script for ultra-lightweight TreeOfLife-10M evaluation.
This script validates the dataset loading without running full CLIP evaluation.
"""

import argparse
import sys
import time

def test_ultralight_config():
    """Test the ultra-lightweight TreeOfLife configuration."""
    print("🚀 Testing Ultra-Lightweight TreeOfLife-10M Configuration")
    print("=" * 60)
    
    try:
        # Test adapter creation
        print("1️⃣  Testing adapter import...")
        from adapters.treeoflife_adapter import TreeOfLifeAdapter
        print("✅ TreeOfLife adapter imported successfully")
        
        # Test adapter with ultra-light settings
        print("\n2️⃣  Testing adapter with ultra-light settings...")
        start_time = time.time()
        
        adapter = TreeOfLifeAdapter(
            root_path='/mnt/d/data',
            split='train_small',
            max_shards=3,  # Only 3 shards (~300MB)
            max_samples=300,  # Only 300 images
            taxonomic_level='family',  # Broader categories
            min_images_per_class=2,
            use_precomputed_embeddings=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Adapter created successfully in {load_time:.2f} seconds")
        print(f"   📊 Dataset size: {len(adapter)} samples")
        print(f"   🏷️  Number of classes: {len(adapter.classes)}")
        print(f"   📝 Sample classes: {adapter.classes[:5]}...")
        
        # Test template loading
        print("\n3️⃣  Testing template loading...")
        templates = adapter.get_templates()
        print(f"✅ Loaded {len(templates)} templates")
        print(f"   📄 First template: '{templates[0]}'")
        
        # Test a single sample
        print("\n4️⃣  Testing sample loading...")
        if len(adapter) > 0:
            sample_image, sample_label = adapter[0]
            print(f"✅ Successfully loaded sample 0")
            print(f"   🖼️  Image shape: {sample_image.size if hasattr(sample_image, 'size') else 'Unknown'}")
            print(f"   🏷️  Label: {sample_label} ({adapter.classes[sample_label]})")
        else:
            print("⚠️  No samples available to test")
        
        print(f"\n🎉 Ultra-lightweight test completed successfully!")
        print(f"⏱️  Total test time: {time.time() - start_time:.2f} seconds")
        print(f"📈 Ready for full evaluation with: python evaluate.py config/treeoflife_ultralight.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test ultra-lightweight TreeOfLife configuration')
    parser.add_argument('--skip-clip', action='store_true', help='Skip CLIP model loading (dataset only)')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples to test')
    
    args = parser.parse_args()
    
    if args.skip_clip:
        print("⚡ Running in skip-CLIP mode (dataset validation only)")
    
    success = test_ultralight_config()
    
    if success:
        print("\n✅ All tests passed! Ready for evaluation.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the configuration.")
        sys.exit(1)

if __name__ == '__main__':
    main()
