#!/usr/bin/env python3
"""Inspect TreeOfLife-10M sample structure to understand available metadata."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inspect_samples():
    """Load and inspect several samples to understand data structure."""
    print("🔍 Inspecting TreeOfLife-10M sample structure...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ HuggingFace datasets library not installed")
        return
    
    # Load the dataset with streaming
    dataset_name = "imageomics/TreeOfLife-10M"
    print(f"🔗 Connecting to {dataset_name}...")
    
    dataset = load_dataset(
        dataset_name, 
        split="train",
        streaming=True,
        cache_dir="/mnt/d/data/treeoflife"
    )
    
    print("✅ Connected! Inspecting first 10 samples...")
    
    for idx, sample in enumerate(dataset):
        if idx >= 10:  # Only inspect first 10 samples
            break
            
        print(f"\n📋 Sample {idx}:")
        print(f"  Keys: {list(sample.keys())}")
        
        # Print detailed info for each field
        for key, value in sample.items():
            if key == 'jpg':
                # Image data - just show type and size info
                print(f"  {key}: {type(value)} (image data)")
            elif key == '__key__':
                print(f"  {key}: {value}")
            elif key == '__url__':
                print(f"  {key}: {value}")
            else:
                # Show other fields with their values (truncated if long)
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")
    
    print("\n🔍 Looking for patterns in __key__ and __url__ fields...")
    
    # Reset iterator and look for patterns in more samples
    dataset = load_dataset(
        dataset_name, 
        split="train",
        streaming=True,
        cache_dir="/mnt/d/data/treeoflife"
    )
    
    keys_seen = []
    urls_seen = []
    
    for idx, sample in enumerate(dataset):
        if idx >= 50:  # Check first 50 samples for patterns
            break
            
        key = sample.get('__key__', '')
        url = sample.get('__url__', '')
        
        keys_seen.append(key)
        urls_seen.append(url)
    
    print(f"\n📊 Sample of __key__ values:")
    for i, key in enumerate(keys_seen[:10]):
        print(f"  {i}: {key}")
    
    print(f"\n📊 Sample of __url__ values:")
    for i, url in enumerate(urls_seen[:10]):
        print(f"  {i}: {url}")
    
    # Look for patterns
    print(f"\n🔍 Analyzing patterns...")
    
    # Check for different sources in URLs
    url_sources = set()
    for url in urls_seen:
        if 'EOL' in url or 'eol' in url.lower():
            url_sources.add('EOL')
        elif 'BIOSCAN' in url or 'bioscan' in url.lower():
            url_sources.add('BIOSCAN')
        elif 'iNaturalist' in url or 'inaturalist' in url.lower():
            url_sources.add('iNaturalist')
        elif 'fishbase' in url.lower():
            url_sources.add('FishBase')
        else:
            url_sources.add('Other')
    
    print(f"📊 URL sources found: {sorted(url_sources)}")
    
    # Check for taxonomic info in keys
    key_patterns = set()
    for key in keys_seen:
        parts = key.split('_')
        if len(parts) > 1:
            key_patterns.add(f"Pattern: {len(parts)} parts")
            if len(parts) <= 3:
                key_patterns.add(f"Example: {key}")
    
    print(f"📊 Key patterns: {sorted(key_patterns)}")

if __name__ == "__main__":
    inspect_samples()
