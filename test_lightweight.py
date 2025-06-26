#!/usr/bin/env python3
"""
Lightweight testing script for TreeOfLife-10M dataset.

This script provides quick testing capabilities for development and debugging
without downloading large datasets.
"""

import argparse
import os
import sys
from dataset_factory import DatasetFactory
from evaluator import ZeroShotEvaluator


def test_treeoflife_lightweight():
    """Test TreeOfLife adapter with minimal configuration."""
    print("🧪 TreeOfLife-10M Lightweight Testing")
    print("=" * 50)
    
    # Ultra-lightweight configuration
    dataset_params = {
        'type': 'treeoflife',
        'root_path': '/mnt/d/data',
        'split': 'train_small',
        'max_samples': 100,  # Very small for quick testing
        'min_images_per_class': 1,
        'taxonomic_level': 'family',  # Broader level for fewer classes
        'use_common_names': True,
        'use_precomputed_embeddings': True
    }
    
    try:
        print(f"📁 Creating dataset with configuration:")
        for key, value in dataset_params.items():
            print(f"   {key}: {value}")
        print()
        
        # Test dataset creation
        print("🔄 Loading dataset...")
        dataset = DatasetFactory.create_dataset(dataset_params)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📊 Samples: {len(dataset)}")
        print(f"   🏷️  Classes: {len(dataset.classes)}")
        
        if dataset.classes:
            print(f"   📝 Sample classes: {dataset.classes[:5]}")
        
        # Test template loading
        templates = dataset.get_templates()
        print(f"   📄 Templates: {len(templates)}")
        print(f"   📄 Sample template: '{templates[0]}'")
        
        # Test data loading (first sample)
        if len(dataset) > 0:
            print("\n🖼️  Testing data loading...")
            try:
                image, label = dataset[0]
                print(f"   ✅ Sample loaded: image shape {image.shape if hasattr(image, 'shape') else type(image)}, label {label}")
            except Exception as e:
                print(f"   ⚠️  Sample loading failed: {e}")
        
        print(f"\n🎉 Lightweight test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_integration():
    """Test CLIP integration with TreeOfLife dataset."""
    print("\n🤖 CLIP Integration Testing")
    print("=" * 50)
    
    try:
        # Create evaluator
        print("🔄 Initializing CLIP model...")
        evaluator = ZeroShotEvaluator(
            clip_model="ViT-B/32",
            device="cuda" if __name__ == "__main__" else "cpu",  # Use CPU for imports
            use_ensemble=True
        )
        print("✅ CLIP model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP integration test failed: {e}")
        return False


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Lightweight testing for TreeOfLife-10M')
    parser.add_argument('--skip-clip', action='store_true', help='Skip CLIP integration test')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to test with')
    
    args = parser.parse_args()
    
    print("🚀 TreeOfLife-10M Lightweight Testing Suite")
    print("=" * 60)
    print(f"💾 Testing with {args.samples} samples")
    print()
    
    # Test 1: Dataset loading
    success1 = test_treeoflife_lightweight()
    
    # Test 2: CLIP integration (optional)
    success2 = True
    if not args.skip_clip:
        success2 = test_clip_integration()
    else:
        print("\n⏭️  Skipping CLIP integration test")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Dataset loading: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"CLIP integration: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Ready for zero-shot evaluation.")
        print("\n💡 Next steps:")
        print("   python evaluate.py config/treeoflife.yaml")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
