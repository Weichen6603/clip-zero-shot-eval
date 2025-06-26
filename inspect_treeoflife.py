#!/usr/bin/env python3
"""Direct dataset inspection for TreeOfLife-10M."""

import time
from datasets import load_dataset

def inspect_dataset():
    """Directly inspect TreeOfLife-10M dataset structure."""
    print("🔍 Direct TreeOfLife-10M dataset inspection")
    print("=" * 50)
    
    try:
        print("📥 Loading dataset with streaming...")
        dataset = load_dataset("imageomics/TreeOfLife-10M", split='train', 
                              cache_dir='/mnt/d/data', streaming=True)
        
        print("✅ Dataset loaded, examining first few samples...")
        
        for i, sample in enumerate(dataset):
            if i >= 3:  # Only look at first 3 samples
                break
                
            print(f"\n🔍 Sample {i}:")
            print(f"   Type: {type(sample)}")
            
            if isinstance(sample, dict):
                print(f"   Keys: {list(sample.keys())}")
                
                # Look for taxonomy-related fields
                taxonomy_keys = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'common']
                print(f"   Taxonomy fields:")
                for key in taxonomy_keys:
                    if key in sample:
                        value = sample[key]
                        print(f"     {key}: {type(value)} = '{value}'")
                    else:
                        print(f"     {key}: NOT FOUND")
                
                # Look for other potentially relevant fields
                other_keys = [k for k in sample.keys() if k not in taxonomy_keys][:10]
                print(f"   Other fields (first 10): {other_keys}")
                
                # Check specific field values
                for key in ['split', 'treeoflife_id', 'image']:
                    if key in sample:
                        value = sample[key]
                        print(f"   {key}: {type(value)} = {str(value)[:100]}")
                        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = inspect_dataset()
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Inspection {'COMPLETED' if success else 'FAILED'} in {elapsed:.1f} seconds")
