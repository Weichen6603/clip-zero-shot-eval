#!/usr/bin/env python3
"""
Script to update TreeOfLife adapter with catalog support.
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def update_adapter():
    """Update the TreeOfLife adapter to use catalog.csv."""
    
    adapter_path = "/home/weichen/Programming/clip-zero-shot-eval/adapters/treeoflife_adapter.py"
    
    # Read the current file
    with open(adapter_path, 'r') as f:
        content = f.read()
    
    # Find the _load_real_data method and replace it
    method_start = content.find("def _load_real_data(self)")
    method_end = content.find("def _process_hf_sample(self", method_start)
    
    if method_start == -1 or method_end == -1:
        print("❌ Could not find method boundaries")
        return False
    
    # Get the indentation level
    lines_before = content[:method_start].split('\n')
    last_line = lines_before[-1] if lines_before else ""
    indent = len(last_line) - len(last_line.lstrip())
    base_indent = " " * indent
    
    # New method implementation
    new_method = f'''{base_indent}def _load_real_data(self) -> List[Dict[str, Any]]:
{base_indent}    """Load real TreeOfLife-10M data from HuggingFace with catalog metadata."""
{base_indent}    try:
{base_indent}        from datasets import load_dataset
{base_indent}        from huggingface_hub import hf_hub_download
{base_indent}    except ImportError:
{base_indent}        raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
{base_indent}    
{base_indent}    print("📥 Loading TreeOfLife-10M dataset from HuggingFace...")
{base_indent}    print("⚠️  This may require authentication: huggingface-cli login")
{base_indent}    
{base_indent}    dataset_name = "imageomics/TreeOfLife-10M"
{base_indent}    
{base_indent}    # Step 1: Download the catalog.csv file first
{base_indent}    print("📋 Downloading catalog.csv for taxonomic labels...")
{base_indent}    try:
{base_indent}        catalog_path = hf_hub_download(
{base_indent}            repo_id=dataset_name,
{base_indent}            filename="metadata/catalog.csv",
{base_indent}            cache_dir=self.root_path,
{base_indent}            repo_type="dataset"
{base_indent}        )
{base_indent}        print(f"✅ Downloaded catalog.csv to: {{catalog_path}}")
{base_indent}        
{base_indent}        # Load catalog into memory
{base_indent}        print("📊 Loading catalog into memory...")
{base_indent}        catalog_df = pd.read_csv(catalog_path)
{base_indent}        print(f"📊 Catalog loaded: {{len(catalog_df)}} entries")
{base_indent}        
{base_indent}        # Create a lookup dictionary for faster access
{base_indent}        self.catalog_lookup = {{}}
{base_indent}        for _, row in catalog_df.iterrows():
{base_indent}            treeoflife_id = row['treeoflife_id']
{base_indent}            self.catalog_lookup[treeoflife_id] = {{
{base_indent}                'kingdom': row.get('kingdom', ''),
{base_indent}                'phylum': row.get('phylum', ''),
{base_indent}                'class': row.get('class', ''),
{base_indent}                'order': row.get('order', ''),
{base_indent}                'family': row.get('family', ''),
{base_indent}                'genus': row.get('genus', ''),
{base_indent}                'species': row.get('species', ''),
{base_indent}                'common': row.get('common', ''),
{base_indent}                'split': row.get('split', 'train')
{base_indent}            }}
{base_indent}        print(f"✅ Created lookup table for {{len(self.catalog_lookup)}} samples")
{base_indent}        
{base_indent}    except Exception as e:
{base_indent}        print(f"⚠️  Could not download catalog.csv: {{e}}")
{base_indent}        print("📡 Falling back to streaming mode with pseudo-random classification...")
{base_indent}        self.catalog_lookup = None
{base_indent}    
{base_indent}    # Step 2: Load the streaming dataset
{base_indent}    print("🔗 Loading streaming dataset...")
{base_indent}    dataset = load_dataset(
{base_indent}        dataset_name, 
{base_indent}        split="train",
{base_indent}        streaming=True,
{base_indent}        cache_dir=self.root_path
{base_indent}    )
{base_indent}    
{base_indent}    print(f"✅ Successfully connected to TreeOfLife-10M dataset")
{base_indent}    print(f"⏱️  Starting data processing...")
{base_indent}    
{base_indent}    # Convert streaming dataset to samples
{base_indent}    samples = []
{base_indent}    processed_count = 0
{base_indent}    start_time = time.time()
{base_indent}    
{base_indent}    print("🔍 Processing samples with real taxonomic labels...")
{base_indent}    
{base_indent}    for idx, sample in enumerate(dataset):
{base_indent}        processed_count += 1
{base_indent}        
{base_indent}        # Check shard limit (approximate)
{base_indent}        if self.max_shards is not None:
{base_indent}            # TreeOfLife-10M has ~140K samples per shard (10M / 73 shards)
{base_indent}            estimated_shard = processed_count // 140000
{base_indent}            if estimated_shard >= self.max_shards:
{base_indent}                print(f"📊 Reached max_shards limit ({{self.max_shards}}), stopping at sample {{processed_count}}")
{base_indent}                break
{base_indent}        
{base_indent}        # Check sample limit
{base_indent}        if self.max_samples is not None and len(samples) >= self.max_samples:
{base_indent}            print(f"📊 Reached max_samples limit ({{self.max_samples}})")
{base_indent}            break
{base_indent}        
{base_indent}        # Extract sample information
{base_indent}        try:
{base_indent}            sample_info = self._process_hf_sample(sample, idx)
{base_indent}            if sample_info:
{base_indent}                samples.append(sample_info)
{base_indent}                
{base_indent}            # Progress indicator (much less verbose)
{base_indent}            if processed_count % 5000 == 0:
{base_indent}                elapsed_time = time.time() - start_time
{base_indent}                print(f"📥 Processed {{processed_count}} samples, found {{len(samples)}} valid samples ({{elapsed_time:.1f}}s)")
{base_indent}                
{base_indent}            # Early exit if we have enough samples for testing
{base_indent}            if len(samples) >= (self.max_samples or 1000):
{base_indent}                print(f"✅ Found enough samples for evaluation ({{len(samples)}}), stopping")
{base_indent}                break
{base_indent}                
{base_indent}        except Exception as e:
{base_indent}            if idx < 3:  # Show details for first few errors
{base_indent}                print(f"⚠️ Error processing sample {{idx}}: {{e}}")
{base_indent}            continue
{base_indent}    
{base_indent}    if not samples:
{base_indent}        print("⚠️ No valid samples found, this might indicate data format issues")
{base_indent}        raise ValueError("No valid samples extracted from TreeOfLife-10M dataset")
{base_indent}    
{base_indent}    elapsed_time = time.time() - start_time
{base_indent}    print(f"✅ Loaded {{len(samples)}} valid samples from TreeOfLife-10M in {{elapsed_time:.1f}}s")
{base_indent}    return self._filter_samples_by_class_count(samples)
{base_indent}        
{base_indent}'''
    
    # Replace the method
    new_content = content[:method_start] + new_method + content[method_end:]
    
    # Write back to file
    with open(adapter_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated TreeOfLife adapter to use catalog.csv")
    return True

if __name__ == "__main__":
    update_adapter()
