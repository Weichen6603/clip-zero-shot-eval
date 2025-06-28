#!/usr/bin/env python3
"""
Test script to verify error handling for corrupted/invalid images.
"""

import io
from PIL import Image
import sys
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

from adapters.treeoflife_adapter import TreeOfLifeAdapter

def test_corrupted_image_handling():
    """Test how the adapter handles corrupted image data."""
    print("🧪 Testing corrupted image handling...")
    
    # Create adapter instance
    adapter = TreeOfLifeAdapter(max_samples=1)
    
    # Test cases for different types of corrupted data
    test_cases = [
        {
            'name': 'Empty bytes',
            'data': b'',
            'should_pass': False
        },
        {
            'name': 'Too small',
            'data': b'abc',
            'should_pass': False
        },
        {
            'name': 'Invalid TIFF header (the error you encountered)',
            'data': b'IIU\x00\x18\x00\x00\x00' + b'invalid_data' * 100,
            'should_pass': False
        },
        {
            'name': 'Fake JPEG header with invalid data',
            'data': b'\xFF\xD8\xFF' + b'not_actually_jpeg_data' * 100,
            'should_pass': False
        },
        {
            'name': 'Valid JPEG signature',
            'data': None,  # Will create actual JPEG
            'should_pass': True
        }
    ]
    
    # Create a valid JPEG for the last test case
    valid_image = Image.new('RGB', (100, 100), (255, 0, 0))
    bytes_io = io.BytesIO()
    valid_image.save(bytes_io, format='JPEG', quality=85)
    test_cases[-1]['data'] = bytes_io.getvalue()
    
    for i, test_case in enumerate(test_cases):
        print(f"\n  Test {i+1}: {test_case['name']}")
        
        # Test _is_valid_image_format
        is_valid_format = adapter._is_valid_image_format(test_case['data'])
        print(f"    Format validation: {'✅ PASS' if is_valid_format == test_case['should_pass'] else '❌ FAIL'}")
        
        # Test actual image loading
        try:
            test_image = Image.open(io.BytesIO(test_case['data']))
            test_image.verify()
            load_success = True
            print(f"    PIL loading: ✅ SUCCESS")
        except Exception as e:
            load_success = False
            print(f"    PIL loading: ❌ FAILED ({type(e).__name__}: {str(e)[:50]}...)")
        
        expected = test_case['should_pass']
        actual = load_success
        result = "✅ EXPECTED" if expected == actual else "⚠️  UNEXPECTED"
        print(f"    Overall: {result} (expected: {expected}, actual: {actual})")

def test_sample_info_validation():
    """Test the sample info validation in the adapter."""
    print("\n🧪 Testing sample info validation...")
    
    adapter = TreeOfLifeAdapter(max_samples=1)
    
    # Simulate corrupted image data in sample_info
    sample_info = {
        'treeoflife_id': 'test_sample',
        'image_source': {
            'type': 'compressed_bytes',
            'image_bytes': b'IIU\x00\x18\x00\x00\x00' + b'corrupted_data' * 50,  # The problematic TIFF header
        }
    }
    
    # Test _load_image_on_demand with corrupted data
    print("  Testing _load_image_on_demand with corrupted TIFF data...")
    result = adapter._load_image_on_demand(sample_info)
    
    if result is None:
        print("  ✅ Correctly rejected corrupted image")
    else:
        print("  ❌ Unexpectedly accepted corrupted image")

if __name__ == "__main__":
    print("🧪 Corrupted Image Handling Test")
    print("=" * 50)
    
    try:
        test_corrupted_image_handling()
        test_sample_info_validation()
        
        print("\n✅ All tests completed!")
        print("💡 The adapter should now handle corrupted images gracefully.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
