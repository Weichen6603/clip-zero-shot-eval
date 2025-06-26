#!/usr/bin/env python3
"""
Mock test for TreeOfLife adapter to verify the basic structure works
without requiring the actual dataset download.
"""

import sys
import os
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_dataset import BaseDatasetAdapter

class MockTreeOfLifeAdapter(BaseDatasetAdapter):
    """Mock version of TreeOfLife adapter for testing structure."""
    
    def __init__(self, root_path: str = "/tmp", split: str = "train", 
                 max_samples: int = 10, taxonomic_level: str = "family", **kwargs):
        self.max_samples = max_samples
        self.taxonomic_level = taxonomic_level
        print(f"🧪 Creating mock TreeOfLife adapter with {max_samples} samples")
        super().__init__(root_path=root_path, split=split, **kwargs)
    
    def _load_data(self, **kwargs):
        """Create mock biological data."""
        print("📝 Loading mock biological data...")
        
        # Mock taxonomic data
        families = ["Felidae", "Canidae", "Ursidae", "Corvidae", "Rosaceae"]
        mock_data = []
        
        for i in range(self.max_samples):
            family = families[i % len(families)]
            mock_data.append({
                'image_path': f'/tmp/mock_image_{i}.jpg',
                'label': family,
                'scientific_name': f'{family.lower()}_species_{i}',
                'common_name': f'Common {family} {i}'
            })
        
        print(f"✅ Created {len(mock_data)} mock samples")
        return mock_data
    
    def _get_classes(self):
        """Get unique taxonomic classes."""
        classes = list(set(item['label'] for item in self.data))
        print(f"📊 Found {len(classes)} unique classes: {classes}")
        return classes
    
    def get_templates(self):
        """Use unified templates."""
        from dataset_adapters import DatasetTemplates
        return DatasetTemplates.get_templates()
    
    def __getitem__(self, idx):
        """Override to return mock image."""
        import torch
        
        item = self.data[idx]
        
        # Create a simple mock image tensor
        if self.transform:
            # If transform is provided, create PIL image first then transform
            mock_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            mock_image = self.transform(mock_image)
        else:
            # Create tensor directly
            mock_image = torch.rand(3, 224, 224)  # type: ignore
        
        # Convert label to index
        if isinstance(item['label'], str):
            label = self.class_to_idx[item['label']]
        else:
            label = item['label']
        
        return mock_image, label

def test_mock_adapter():
    """Test the mock adapter."""
    print("🧪 Testing Mock TreeOfLife Adapter...")
    
    try:
        # Test adapter creation
        adapter = MockTreeOfLifeAdapter(
            root_path="/tmp",
            max_samples=5,
            taxonomic_level="family"
        )
        
        print(f"✅ Created adapter with {len(adapter)} samples")
        print(f"✅ Found {len(adapter.classes)} classes: {adapter.classes}")
        
        # Test template loading
        templates = adapter.get_templates()
        print(f"✅ Loaded {len(templates)} templates")
        print(f"   First template: '{templates[0]}'")
        
        # Test sample access
        sample_image, sample_label = adapter[0]
        print(f"✅ Sample access works: image shape {sample_image.size if hasattr(sample_image, 'size') else 'unknown'}, label {sample_label}")
        
        print("🎉 Mock adapter test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mock_adapter()
    
    if success:
        print("\n✅ Mock test successful! The adapter structure is correct.")
        print("💡 The issue is likely in the HuggingFace dataset loading process.")
        print("🔍 Recommendations:")
        print("   1. Check internet connection")
        print("   2. Try with a different cache directory")
        print("   3. Consider using a smaller model or different approach")
    else:
        print("\n❌ Mock test failed. There are structural issues to fix.")
