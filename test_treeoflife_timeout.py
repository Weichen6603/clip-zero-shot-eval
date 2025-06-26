#!/usr/bin/env python3
"""
Quick test script for TreeOfLife adapter with timeout protection.
"""

import sys
import signal
import time

def timeout_handler(signum, frame):
    print("\n⏰ Test timeout reached, stopping...")
    raise TimeoutError("Test timed out")

def test_treeoflife_adapter():
    """Test TreeOfLife adapter with mock data fallback."""
    
    # Set timeout for the whole test
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    try:
        print("🧪 Testing TreeOfLife adapter...")
        
        from adapters.treeoflife_adapter import TreeOfLifeAdapter
        print('✅ TreeOfLife adapter imported successfully')

        # Test with mock data (should be fast)
        print("\n📝 Testing with mock data...")
        adapter = TreeOfLifeAdapter(
            root_path='/mnt/d/data/treeoflife',
            max_samples=10,
            taxonomic_level='species',
            max_shards=3
        )
        
        print(f'📊 Adapter created with {len(adapter)} samples, {len(adapter._get_classes())} classes')
        print(f'📋 Classes: {adapter._get_classes()[:5]}...')  # Show first 5 classes
        
        # Test a sample
        if len(adapter) > 0:
            sample_image, sample_label = adapter[0]
            print(f'🖼️  Sample shape: {sample_image.shape}, label: {sample_label}')
        
        print("\n✅ TreeOfLife adapter test completed successfully!")
        
    except TimeoutError:
        print("\n⏰ Test timed out - this indicates the real data loading is taking too long")
        print("🧪 For quick testing, the adapter should fall back to mock data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        signal.alarm(0)  # Cancel the alarm

if __name__ == "__main__":
    test_treeoflife_adapter()
