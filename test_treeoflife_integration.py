#!/usr/bin/env python3
"""
Test script for TreeOfLife adapter integration.
Tests both database creation and real data loading.
"""

import sys
import os
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

from adapters.treeoflife_adapter import TreeOfLifeAdapter
import torch

def test_treeoflife_adapter():
    """Test TreeOfLife adapter with minimal configuration."""
    print("🧪 Testing TreeOfLife adapter...")
    
    # Test configuration
    config = {
        'root_path': '/mnt/d/data/treeoflife',
        'taxonomic_level': 'class',
        'max_shards': 1,  # Use only 1 shard for testing
        'max_samples': 5,  # Load only 5 samples for testing
        'exclude_partial_labels': False
    }
    
    print(f"📋 Test configuration: {config}")
    
    # Initialize adapter
    try:
        adapter = TreeOfLifeAdapter(**config)
        print(f"✅ Adapter initialized successfully")
        print(f"📊 Dataset size: {len(adapter)}")
        print(f"🏷️  Number of classes: {len(adapter.classes)}")
        print(f"🔤 Classes: {adapter.classes[:10]}...")  # Show first 10 classes
        
        # Test data access
        if len(adapter) > 0:
            print("\n🔍 Testing sample access...")
            for i in range(min(3, len(adapter))):
                try:
                    image, label = adapter[i]
                    class_name = adapter.classes[label]
                    
                    print(f"  Sample {i}: image shape={image.shape}, label={label}, class='{class_name}'")
                    
                    # Access the underlying sample info
                    sample_info = adapter.data[i]
                    print(f"    📋 Sample ID: {sample_info['treeoflife_id']}")
                    print(f"    🧬 Scientific name: {sample_info['scientific_name']}")
                    print(f"    🏷️  Common name: {sample_info['common_name']}")
                    
                except Exception as e:
                    print(f"  ❌ Error accessing sample {i}: {e}")
        
        print("\n✅ TreeOfLife adapter test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_creation():
    """Test the SQLite database creation process."""
    print("\n🗄️  Testing database creation...")
    
    # Check if catalog.csv exists
    catalog_path = "/mnt/d/data/treeoflife/datasets--imageomics--TreeOfLife-10M/snapshots/*/metadata/catalog.csv"
    import glob
    catalog_files = glob.glob(catalog_path)
    
    if not catalog_files:
        print(f"⚠️  No catalog.csv found at {catalog_path}")
        return False
    
    catalog_file = catalog_files[0]
    print(f"📋 Found catalog at: {catalog_file}")
    
    # Check file size
    file_size = os.path.getsize(catalog_file) / (1024**3)  # GB
    print(f"📊 Catalog size: {file_size:.2f} GB")
    
    # Test database creation
    from adapters.treeoflife_adapter import TreeOfLifeAdapter
    adapter = TreeOfLifeAdapter(
        root_path='/mnt/d/data/treeoflife',
        max_shards=1,  # Load 1 shard for testing
        max_samples=5  # Load just 5 samples
    )
    
    try:
        adapter._init_catalog_database(catalog_file)
        
        # Check if database was created
        db_path = catalog_file.replace('.csv', '.db')
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path) / (1024**2)  # MB
            print(f"✅ Database created: {db_path} ({db_size:.1f} MB)")
            
            # Test a lookup
            taxonomy = adapter._get_taxonomy_from_catalog("sample_12345")
            print(f"🔍 Test lookup result: {taxonomy}")
            
            return True
        else:
            print(f"❌ Database not found at {db_path}")
            return False
            
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting TreeOfLife integration tests...\n")
    
    # Test 1: Database creation
    db_success = test_database_creation()
    
    # Test 2: Adapter functionality
    adapter_success = test_treeoflife_adapter()
    
    print(f"\n📊 Test Results:")
    print(f"  🗄️  Database creation: {'✅ PASS' if db_success else '❌ FAIL'}")
    print(f"  🔧 Adapter functionality: {'✅ PASS' if adapter_success else '❌ FAIL'}")
    
    if db_success and adapter_success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
