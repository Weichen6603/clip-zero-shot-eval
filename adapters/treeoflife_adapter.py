"""TreeOfLife-10M dataset adapter with HuggingFace integration for efficient loading."""

import os
import sys
import time
import random
import pickle
import warnings
import io
import traceback
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from PIL import Image, UnidentifiedImageError
import torch
import pandas as pd
import torchvision.transforms as transforms

# Configure PIL to handle large images and suppress warnings
Image.MAX_IMAGE_PIXELS = None  # Remove size limit
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message=".*TIFF.*")  # Ignore TIFF warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class TreeOfLifeAdapter(BaseDatasetAdapter):
    """
    TreeOfLife-10M dataset adapter supporting HuggingFace streaming and Pandas+local images.
    
    Dataset splits:
    - train: ~9.5M samples
    - train_small: ~953K samples (10% subset for faster experimentation)
    - val: ~502K samples
    
    Features:
    - HuggingFace streaming for efficient data loading
    - Pandas+local images support for train_small quick experiments
    - Flexible taxonomic level selection (species, genus, family, etc.)
    - SPEED OPTIMIZATIONS:
      - Memory-cached catalog for ultra-fast lookups
      - Batch processing for reduced overhead
      - Smart sampling strategies
    """
    
    def __init__(self, root_path: str = "./data/treeoflife", transform=None, split: str = "train", 
                 max_samples: Optional[int] = None, taxonomic_level: str = "species",
                 min_images_per_class: int = 1, exclude_partial_labels: bool = False,
                 strict_label_filtering: bool = False, 
                 images_dir: Optional[str] = None, 
                 catalog_cache_size: int = 50000, use_smart_sampling: bool = True, 
                 streaming: bool = True, **kwargs):
        """
        Initialize TreeOfLife-10M adapter.
        
        Args:
            root_path: Path to dataset cache directory (default: ./data/treeoflife)
            transform: Image transformations to apply
            split: Dataset split ('train', 'train_small', 'val')
            max_samples: Maximum number of samples to use (None for all)
            taxonomic_level: Level of taxonomy to use for classification 
                           ('species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom')
            min_images_per_class: Minimum images per class to include class
            exclude_partial_labels: Whether to exclude images without full taxonomic labels
            strict_label_filtering: Whether to apply strict filtering of uncertain/confusing labels
            images_dir: Local images directory (default: root_path/images)
            catalog_cache_size: Size of in-memory catalog cache (default: 50000)
            use_smart_sampling: Whether to use smart sampling for faster train_small loading
            streaming: Whether to use HuggingFace streaming mode (default: True)
        """
        self.max_samples = max_samples
        self.taxonomic_level = taxonomic_level.lower()
        self.min_images_per_class = min_images_per_class
        self.exclude_partial_labels = exclude_partial_labels
        self.strict_label_filtering = strict_label_filtering
        self.images_dir = images_dir or os.path.join(root_path, "images")
        self.split = split
        self.root_path = root_path
        self.catalog_cache = {}  # Initialize cache for catalog lookups
        self.catalog_cache_size = catalog_cache_size
        self.use_smart_sampling = use_smart_sampling
        self.streaming = streaming
        
        # New optimization attributes
        self.memory_catalog = None  # Full in-memory catalog for train_small
        self.split_sample_ids = None  # Pre-filtered sample IDs for target split
        
        # Validate taxonomic level
        valid_levels = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
        if self.taxonomic_level not in valid_levels:
            raise ValueError(f"taxonomic_level must be one of {valid_levels}")
        
        # Validate split
        valid_splits = ['train', 'train_small', 'val']
        if self.split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")
        
        # Call parent constructor with required parameters
        super().__init__(root_path=root_path, transform=transform, split=split, **kwargs)
        
    def _ensure_catalog_csv(self):
        """Ensure catalog.csv exists locally. Download from HuggingFace if missing."""
        import os, shutil
        catalog_dir = os.path.join(self.root_path, "metadata")
        catalog_path = os.path.join(catalog_dir, "catalog.csv")
        if not os.path.exists(catalog_path):
            print(f"📥 catalog.csv not found, downloading...")
            os.makedirs(catalog_dir, exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                dataset_name = "imageomics/TreeOfLife-10M"
                downloaded_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename="metadata/catalog.csv",
                    cache_dir=self.root_path,
                    repo_type="dataset"
                )
                print(f"✅ catalog.csv downloaded to cache: {downloaded_path}")
                shutil.copy(downloaded_path, catalog_path)
                print(f"✅ catalog.csv copied to local metadata: {catalog_path}")
            except Exception as e:
                print(f"❌ Failed to download catalog.csv: {e}")
                raise FileNotFoundError(f"catalog.csv not found and download failed: {e}")
        return catalog_path

    def _load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset using either local files or online streaming."""
        print(f"🌳 Loading TreeOfLife-10M data...")
        print(f"    Split: {self.split}")
        print(f"    Streaming: {self.streaming}")
        print(f"    Max samples: {self.max_samples}")
        print(f"    Taxonomic level: {self.taxonomic_level}")
        
        self.catalog_path = self._ensure_catalog_csv()
        
        if self.streaming:
            # 在线模式：使用随机采样
            print("📡 Using online streaming mode")
            print("⚠️  Note: Due to ID mismatch, will use random sampling")
            return self._load_streaming_with_random_sampling()
        else:
            # 本地模式：严格匹配 catalog
            print("💾 Using local mode")
            
            # 预加载 catalog 用于精确匹配
            print("🚀 Pre-loading catalog for exact matching...")
            self._preload_catalog_for_split()
            
            # 检查本地图片
            if self._check_local_images():
                print("✅ Local images found, using catalog-based matching")
                return self._load_local_with_catalog_matching()
            else:
                print("📥 No local images found. Need to download first.")
                print("\n📋 Download instructions:")
                print("1. Download the dataset with correct treeoflife_id naming")
                print("2. Place images in: " + self.images_dir)
                print("3. Ensure files are named: {treeoflife_id}.jpg")
                print("\n💡 Tip: Use 'streaming: true' for online mode instead")
                raise FileNotFoundError(f"No local images found in {self.images_dir}")
    
    def _load_streaming_with_random_sampling(self) -> List[Dict[str, Any]]:
        """Load data using streaming with random sampling to match split sizes."""
        print("📥 Loading TreeOfLife-10M with random sampling...")
        
        try:
            from datasets import load_dataset
            import random
        except ImportError:
            raise ImportError("Required: pip install datasets")
        
        # Define target sizes for each split
        split_sizes = {
            'train': 9533174,
            'train_small': 953202,
            'val': 501656
        }
        
        target_size = split_sizes.get(self.split, 953202)
        if self.max_samples:
            target_size = min(self.max_samples, target_size)
        
        # Calculate sampling rate
        total_dataset_size = 10988032  # ~11M total samples
        sampling_rate = target_size / total_dataset_size
        
        print(f"🎯 Target: {target_size:,} samples")
        print(f"📊 Sampling rate: {sampling_rate:.2%}")
        
        try:
            dataset = load_dataset(
                "imageomics/TreeOfLife-10M",
                split="train",
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return []
        
        samples = []
        processed_count = 0
        start_time = time.time()
        
        from tqdm import tqdm
        pbar = tqdm(total=target_size, desc=f"Sampling {self.split}", unit="samples")
        
        for sample in dataset:
            processed_count += 1
            
            # Random sampling based on split
            if random.random() > sampling_rate:
                continue
            
            try:
                # Get sample ID (use __key__ since treeoflife_id not available)
                sample_id = sample.get('__key__', f'sample_{processed_count}')
                
                # Get image data
                image_data = sample.get('jpg') or sample.get('image')
                if not image_data:
                    continue
                
                # Process image
                image_bytes = self._compress_image_fast(image_data, sample_id)
                if not image_bytes:
                    continue
                
                # Create placeholder taxonomy (since we can't match with catalog)
                # You can implement more sophisticated labeling here
                taxonomic_label = self._generate_placeholder_label(len(samples))
                
                samples.append({
                    'index': len(samples),
                    'taxonomic_label': taxonomic_label,
                    'treeoflife_id': sample_id,
                    'image_bytes': image_bytes,
                    'image_path': None,
                    'metadata': {
                        'split': self.split,
                        'streaming_id': sample_id
                    }
                })
                
                pbar.update(1)
                
                if len(samples) >= target_size:
                    break
                    
            except Exception as e:
                if processed_count <= 10:
                    print(f"\n⚠️ Error: {type(e).__name__}")
                continue
        
        pbar.close()
        
        elapsed = time.time() - start_time
        print(f"\n📊 Streaming Statistics:")
        print(f"  ⏱️  Time: {elapsed:.1f}s")
        print(f"  📝 Processed: {processed_count:,} samples")
        print(f"  ✅ Sampled: {len(samples):,} samples")
        print(f"  🚀 Speed: {processed_count/elapsed:.1f} samples/sec")
        
        if self.min_images_per_class > 1:
            print("\n⚠️  Note: Cannot apply min_images_per_class filter with placeholder labels")
        
        return samples
    
    def _load_local_with_catalog_matching(self) -> List[Dict[str, Any]]:
        """Load local images with strict catalog matching."""
        print("📋 Loading with exact catalog matching...")
        
        if not self.memory_catalog:
            raise RuntimeError("Catalog not loaded properly")
        
        samples = []
        missing_images = 0
        
        from tqdm import tqdm
        
        # Iterate through catalog entries for this split
        for sample_id, metadata in tqdm(self.memory_catalog.items(), 
                                       desc=f"Loading {self.split}", 
                                       total=len(self.memory_catalog)):
            
            # Construct image path
            image_path = os.path.join(self.images_dir, f"{sample_id}.jpg")
            
            # Check if image exists
            if not os.path.exists(image_path):
                missing_images += 1
                continue
            
            # Get taxonomic label
            label = metadata.get(self.taxonomic_level)
            if not label or not self._is_valid_label(label):
                continue
            
            # Check partial labels
            if self.exclude_partial_labels and not self._has_full_taxonomy_from_dict(metadata):
                continue
            
            samples.append({
                'index': len(samples),
                'taxonomic_label': label,
                'treeoflife_id': sample_id,
                'image_path': image_path,
                'image_bytes': None,
                'metadata': metadata
            })
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        print(f"\n📊 Local Loading Statistics:")
        print(f"  📁 Catalog entries: {len(self.memory_catalog):,}")
        print(f"  ✅ Images found: {len(samples):,}")
        print(f"  ❌ Missing images: {missing_images:,}")
        
        return self._filter_samples_by_class_count(samples)
    
    def _generate_placeholder_label(self, index: int) -> str:
        """Generate placeholder labels for streaming mode."""
        # Simple placeholder - you can make this more sophisticated
        # For example, use clustering on image features
        return f"class_{index % 1000}"
    
    def _load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset using either local files or online streaming."""
        print(f"🌳 Loading TreeOfLife-10M data...")
        print(f"    Split: {self.split}")
        print(f"    Streaming: {self.streaming}")
        print(f"    Max samples: {self.max_samples}")
        print(f"    Taxonomic level: {self.taxonomic_level}")
        
        self.catalog_path = self._ensure_catalog_csv()
        
        if self.streaming:
            # 在线模式：使用随机采样
            print("📡 Using online streaming mode")
            print("⚠️  Note: Due to ID mismatch, will use random sampling")
            return self._load_streaming_with_random_sampling()
        else:
            # 本地模式：严格匹配 catalog
            print("💾 Using local mode")
            
            # 预加载 catalog 用于精确匹配
            print("🚀 Pre-loading catalog for exact matching...")
            self._preload_catalog_for_split()
            
            # 检查本地图片
            if self._check_local_images():
                print("✅ Local images found, using catalog-based matching")
                return self._load_local_with_catalog_matching()
            else:
                print("📥 No local images found.")
                
                # 提供选项
                print("\n🔧 Options:")
                print("1. Use streaming mode instead (set 'streaming: true')")
                print("2. Download specific split from HuggingFace")
                print("3. Use pre-downloaded images")
                
                # 尝试智能下载
                if self.split == "train_small":
                    print("\n💡 For train_small, consider downloading the subset:")
                    self._provide_download_instructions()
                else:
                    print(f"\n💡 For {self.split}, downloading {self._get_split_size():,} images would be required")
                
                raise FileNotFoundError(
                    f"No local images found in {self.images_dir}. "
                    f"Use 'streaming: true' for online mode or download images first."
                )
    
    def _get_split_size(self) -> int:
        """Get the size of current split."""
        split_sizes = {
            'train': 9533174,
            'train_small': 953202,
            'val': 501656
        }
        return split_sizes.get(self.split, 0)
    
    def _provide_download_instructions(self):
        """Provide specific download instructions for each split."""
        print("\n📋 Download Instructions:")
        print("=" * 60)
        
        if self.split == "train_small":
            print("Option 1: Download pre-packaged train_small (if available)")
            print("```bash")
            print("# Check if train_small subset is available separately")
            print("# on HuggingFace or other sources")
            print("```")
        
        print("\nOption 2: Extract from streaming (recommended for train_small)")
        print("```python")
        print("# save_train_small_locally.py")
        print("from datasets import load_dataset")
        print("import os")
        print("from PIL import Image")
        print("from tqdm import tqdm")
        print()
        print(f"output_dir = '{self.images_dir}'")
        print("os.makedirs(output_dir, exist_ok=True)")
        print()
        print("# Load dataset in streaming mode")
        print("ds = load_dataset('imageomics/TreeOfLife-10M', split='train', streaming=True)")
        print()
        print("# Load catalog to get train_small IDs")
        print(f"import pandas as pd")
        print(f"catalog = pd.read_csv('{self.catalog_path}')")
        print(f"target_ids = set(catalog[catalog['split'] == '{self.split}']['treeoflife_id'])")
        print(f"print(f'Target: {{len(target_ids):,}} images')")
        print()
        print("# Extract and save images")
        print("saved = 0")
        print("for sample in tqdm(ds, desc='Scanning'):")
        print("    # Since IDs don't match, we'll save first N images")
        print(f"    if saved >= {self._get_split_size()}:")
        print("        break")
        print("    ")
        print("    image = sample.get('jpg') or sample.get('image')")
        print("    if image:")
        print("        # Use sequential naming or hash")
        print("        filename = f'{saved:08d}.jpg'")
        print("        image.save(os.path.join(output_dir, filename))")
        print("        saved += 1")
        print("```")
        
        print("\nOption 3: Use external download tools")
        print("- Check TreeOfLife project website")
        print("- Look for direct download links")
        print("- Contact dataset maintainers")
        
        print("\n⚠️  Note: Due to ID mismatch between HuggingFace and catalog,")
        print("    exact train_small extraction is challenging")
    
    def _smart_download_attempt(self) -> bool:
        """Attempt smart download for specific scenarios."""
        print("\n🤖 Attempting smart download...")
        
        # For train_small, we could try to download a subset
        if self.split == "train_small" and self._get_split_size() < 1_000_000:
            response = input("\n❓ Download ~1M images for train_small? This may take a while [y/N]: ")
            if response.lower() == 'y':
                return self._download_split_subset()
        
        return False
    
    def _download_split_subset(self) -> bool:
        """Download a subset of images for the current split."""
        try:
            from datasets import load_dataset
            from tqdm import tqdm
            import hashlib
            
            print(f"\n📥 Downloading {self.split} subset...")
            os.makedirs(self.images_dir, exist_ok=True)
            
            # Since we can't match IDs, we'll create a mapping file
            mapping_file = os.path.join(self.root_path, f"{self.split}_id_mapping.json")
            mapping = {}
            
            # Load streaming dataset
            ds = load_dataset("imageomics/TreeOfLife-10M", split="train", streaming=True)
            
            target_size = self._get_split_size()
            saved_count = 0
            
            # Use deterministic sampling for reproducibility
            random.seed(42)
            sampling_rate = target_size / 10_988_032  # Total dataset size
            
            print(f"📊 Sampling rate: {sampling_rate:.2%}")
            
            for idx, sample in enumerate(tqdm(ds, desc="Downloading", total=target_size)):
                # Deterministic sampling
                if random.random() > sampling_rate:
                    continue
                
                # Get image
                image = sample.get('jpg') or sample.get('image')
                if not image:
                    continue
                
                # Generate consistent filename
                hf_id = sample.get('__key__', f'sample_{idx}')
                # Create a deterministic ID based on content
                stable_id = hashlib.md5(f"{self.split}_{saved_count}".encode()).hexdigest()
                
                # Save image
                filename = f"{stable_id}.jpg"
                filepath = os.path.join(self.images_dir, filename)
                
                if hasattr(image, 'save'):
                    image.save(filepath, 'JPEG')
                else:
                    with open(filepath, 'wb') as f:
                        f.write(image)
                
                # Save mapping
                mapping[stable_id] = {
                    'hf_key': hf_id,
                    'index': saved_count
                }
                
                saved_count += 1
                if saved_count >= target_size:
                    break
            
            # Save mapping
            import json
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f)
            
            print(f"\n✅ Downloaded {saved_count:,} images")
            print(f"📄 Mapping saved to: {mapping_file}")
            
            return saved_count > 0
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False
    
    def _generate_download_script(self) -> str:
        """Generate a Python script to download train_small images."""
        return '''#!/usr/bin/env python3
"""
Download TreeOfLife-10M train_small images
"""

import os
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
import tarfile
import shutil

def download_train_small():
    """Download and extract train_small images."""
    
    # Setup paths
    root_path = "./data/treeoflife"
    images_dir = os.path.join(root_path, "images")
    catalog_path = os.path.join(root_path, "metadata", "catalog.csv")
    
    os.makedirs(images_dir, exist_ok=True)
    
    print("📋 Loading catalog to identify train_small images...")
    df = pd.read_csv(catalog_path)
    train_small_ids = set(df[df['split'] == 'train_small']['treeoflife_id'].tolist())
    print(f"✅ Found {len(train_small_ids):,} train_small images to download")
    
    # Option 1: Download full dataset and extract only train_small
    # This requires knowing which tar files contain which images
    
    print("\\n⚠️  Manual steps required:")
    print("1. Download image tar files from HuggingFace")
    print("2. Extract only the train_small images based on catalog.csv")
    print("3. Place them in: " + images_dir)
    print("4. Name format: {treeoflife_id}.jpg")
    
    # TODO: Implement actual download logic based on dataset structure

if __name__ == "__main__":
    download_train_small()
'''
    
    def _check_local_images(self):
        """Check if local images exist for the current split."""
        try:
            # Check images directory
            if os.path.exists(self.images_dir):
                jpg_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')]
                if jpg_files:
                    print(f"✅ Found {len(jpg_files):,} local images in {self.images_dir}")
                    return True
                else:
                    print(f"📁 Images directory exists but no .jpg files found in {self.images_dir}")
            else:
                print(f"📁 Images directory not found: {self.images_dir}")
        except Exception as e:
            print(f"⚠️ Error checking local images: {e}")
        return False
    
    def _preload_catalog_for_split(self):
        """OPTIMIZATION: Pre-load only the relevant catalog data into memory."""
        print("🔄 Pre-loading catalog data into memory...")
        
        # Check for cached memory catalog
        cache_file = os.path.join(self.root_path, f"catalog_cache_{self.split}.pkl")
        
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(self.catalog_path):
            print("⚡ Loading cached catalog from disk...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.memory_catalog = cache_data['memory_catalog']
                    self.split_sample_ids = cache_data['split_sample_ids']
                print(f"✅ Loaded {len(self.memory_catalog):,} catalog entries from cache")
                return
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}, rebuilding...")
        
        # Build memory catalog
        print("🔧 Building in-memory catalog...")
        self.memory_catalog = {}
        self.split_sample_ids = set()  # Use set for O(1) lookups
        
        # Read catalog with pandas for speed
        try:
            import pandas as pd
            print("📊 Reading catalog with pandas...")
            
            # Read only necessary columns to save memory
            usecols = ['split', 'treeoflife_id', 'kingdom', 'phylum', 'class', 
                      'order', 'family', 'genus', 'species', 'common']
            df = pd.read_csv(self.catalog_path, usecols=usecols, dtype=str)  # Use str to avoid mixed types
            
            print(f"📊 Catalog loaded: {len(df):,} total entries")
            
            # Filter by split
            df_split = df[df['split'] == self.split].copy()
            print(f"📊 Filtered to {self.split}: {len(df_split):,} entries")
            
            # Convert to dictionary for fast lookups
            for _, row in df_split.iterrows():
                sample_id = str(row['treeoflife_id']).strip()
                if not sample_id:
                    continue
                    
                self.memory_catalog[sample_id] = {
                    'kingdom': str(row.get('kingdom', '')).strip() if pd.notna(row.get('kingdom')) else '',
                    'phylum': str(row.get('phylum', '')).strip() if pd.notna(row.get('phylum')) else '',
                    'class': str(row.get('class', '')).strip() if pd.notna(row.get('class')) else '',
                    'order': str(row.get('order', '')).strip() if pd.notna(row.get('order')) else '',
                    'family': str(row.get('family', '')).strip() if pd.notna(row.get('family')) else '',
                    'genus': str(row.get('genus', '')).strip() if pd.notna(row.get('genus')) else '',
                    'species': str(row.get('species', '')).strip() if pd.notna(row.get('species')) else '',
                    'common': str(row.get('common', '')).strip() if pd.notna(row.get('common')) else '',
                    'split': self.split
                }
                self.split_sample_ids.add(sample_id)
            
            print(f"✅ Built memory catalog: {len(self.memory_catalog):,} entries")
            
            # Cache to disk
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data = {
                    'memory_catalog': self.memory_catalog,
                    'split_sample_ids': self.split_sample_ids
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 Cached catalog to: {cache_file}")
            except Exception as e:
                print(f"⚠️ Failed to cache catalog: {e}")
                
        except Exception as e:
            print(f"⚠️ Failed to pre-load catalog with pandas: {e}")
            print("📊 Catalog preloading failed, will use slower lookups...")
            # Don't use text index fallback for train_small
            self.memory_catalog = None
            self.split_sample_ids = None
    
    def _init_catalog_lazy_reader(self, catalog_path: str):
        """Initialize ultra-lightweight catalog reader using text-based indexing."""
        self.catalog_path = catalog_path
        self.catalog_cache = {}  # Minimal cache for recently accessed entries
        
        # Create a simple text index if it doesn't exist
        index_path = catalog_path.replace('.csv', '_simple_index.txt')
        
        if not os.path.exists(index_path) or os.path.getmtime(index_path) < os.path.getmtime(catalog_path):
            print("🔧 Creating ultra-lightweight text index...")
            self._create_simple_index(catalog_path, index_path)
        else:
            print("✅ Using existing text index")
        
        self.index_path = index_path
        print(f"✅ Ultra-lightweight catalog reader initialized")
    
    def _create_simple_index(self, catalog_path: str, index_path: str):
        """Create minimal index mapping sample_id to line number."""
        print("📝 Building minimal index (this will take a few minutes but only once)...")
        
        with open(index_path, 'w') as index_file:
            with open(catalog_path, 'r', encoding='utf-8') as csv_file:
                # Skip header and get column positions
                header_line = csv_file.readline()
                header = header_line.strip().split(',')
                
                # Find treeoflife_id column
                try:
                    id_col = header.index('treeoflife_id')
                except ValueError:
                    id_col = 1  # Default assumption
                
                line_num = 1  # Start from 1 (after header)
                entries_indexed = 0
                
                for line in csv_file:
                    line_num += 1
                    try:
                        # Split line and extract ID (handle quoted fields)
                        parts = self._parse_csv_line(line)
                        
                        if len(parts) > id_col:
                            sample_id = parts[id_col].strip()
                            if sample_id and sample_id != 'treeoflife_id':
                                index_file.write(f"{sample_id}:{line_num}\n")
                                entries_indexed += 1
                                
                                if entries_indexed % 200000 == 0:
                                    print(f"📊 Indexed {entries_indexed:,} entries...")
                    
                    except Exception:
                        continue  # Skip malformed lines
                
                print(f"✅ Index complete: {entries_indexed:,} entries")
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse a CSV line manually to extract fields."""
        # Simple CSV parsing (handles basic quoting)
        parts = []
        current_part = ""
        in_quotes = False
        
        for char in line.strip():
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip('"'))
                current_part = ""
                continue
            current_part += char
        
        if current_part:
            parts.append(current_part.strip('"'))
        
        return parts
    
    def _load_pandas_data(self) -> List[Dict[str, Any]]:
        """Load data using Pandas and local images."""
        import pandas as pd
        from tqdm import tqdm
        
        print(f"📋 Reading catalog.csv: {self.catalog_path}")
        df = pd.read_csv(self.catalog_path)
        
        # Filter by split
        df_split = df[df["split"] == self.split].copy()
        print(f"✅ {self.split} sample count: {len(df_split):,}")
        
        # Apply max_samples if specified
        if self.max_samples:
            df_split = df_split.sample(n=min(self.max_samples, len(df_split)), random_state=42)
            print(f"📊 Sampled {len(df_split):,} samples")
        
        # Build image paths
        def get_image_path(treeoflife_id):
            return os.path.join(self.images_dir, f"{treeoflife_id}.jpg")
        
        df_split["image_path"] = df_split["treeoflife_id"].apply(get_image_path)
        
        # Select label based on taxonomic level
        if self.taxonomic_level not in df_split.columns:
            print(f"⚠️ Taxonomic level '{self.taxonomic_level}' not found, using 'species'")
            label_col = "species"
        else:
            label_col = self.taxonomic_level
        
        # Build samples
        samples = []
        skipped_no_label = 0
        skipped_no_image = 0
        
        for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc="Building samples"):
            # Get label
            label = row[label_col] if pd.notna(row[label_col]) else None
            if not label or str(label).strip() == "":
                skipped_no_label += 1
                continue
            
            # Apply strict filtering if enabled
            if self.strict_label_filtering:
                label_str = str(label).strip()
                if not self._is_valid_label(label_str):
                    skipped_no_label += 1
                    continue
            
            # Check if excluding partial labels
            if self.exclude_partial_labels:
                if not self._has_full_taxonomy(row):
                    skipped_no_label += 1
                    continue
            
            # Check image exists
            image_path = row["image_path"]
            if not os.path.exists(image_path):
                skipped_no_image += 1
                continue
            
            # Create metadata dict
            metadata = {
                'kingdom': str(row.get('kingdom', '')) if pd.notna(row.get('kingdom')) else '',
                'phylum': str(row.get('phylum', '')) if pd.notna(row.get('phylum')) else '',
                'class': str(row.get('class', '')) if pd.notna(row.get('class')) else '',
                'order': str(row.get('order', '')) if pd.notna(row.get('order')) else '',
                'family': str(row.get('family', '')) if pd.notna(row.get('family')) else '',
                'genus': str(row.get('genus', '')) if pd.notna(row.get('genus')) else '',
                'species': str(row.get('species', '')) if pd.notna(row.get('species')) else '',
                'common': str(row.get('common', '')) if pd.notna(row.get('common')) else '',
            }
            
            samples.append({
                "index": len(samples),
                "taxonomic_label": str(label).strip(),
                "treeoflife_id": row["treeoflife_id"],
                "image_path": image_path,
                "image_bytes": None,  # No bytes in local mode
                "metadata": metadata,
            })
        
        print(f"✅ Valid samples: {len(samples):,}")
        print(f"⚠️ Skipped - no valid label: {skipped_no_label:,}")
        print(f"⚠️ Skipped - no image file: {skipped_no_image:,}")
        
        return self._filter_samples_by_class_count(samples)

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        # Based on the actual catalog.csv analysis
        split_sizes = {
            'train': 9533174,
            'train_small': 953202,
            'val': 501656
        }
        return split_sizes.get(self.split, 10_000_000)
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """Load TreeOfLife-10M using streaming without ID filtering."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        print(f"🔗 Loading streaming dataset...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return []

        samples = []
        processed_count = 0
        failed_images = []
        start_time = time.time()
        
        print("🔍 Processing samples...")
        print("⚠️  Note: Cannot pre-filter train_small due to ID mismatch")
        print("📊 Will process all samples and filter by metadata")
        
        # Target number depends on whether we're filtering
        if self.split == "train_small":
            # We need to process many more samples to find all train_small ones
            estimated_to_process = 10_000_000  # Full dataset
            target_samples = 953_202  # train_small size
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            estimated_to_process = target_samples
        
        if self.max_samples:
            target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} valid samples")
        else:
            print(f"🎯 Target: {target_samples:,} {self.split} samples")

        from tqdm import tqdm
        pbar = tqdm(total=target_samples, desc="Finding samples", unit="samples")
        
        # Process samples
        for sample in dataset:
            if len(samples) >= target_samples:
                print(f"\n🏁 Found target number of samples. Stopping.")
                break

            processed_count += 1
            
            # Periodic progress update
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                found_rate = len(samples) / processed_count if processed_count > 0 else 0
                pbar.set_postfix({
                    'processed': f'{processed_count:,}',
                    'rate': f'{rate:.1f}/s',
                    'found_rate': f'{found_rate:.2%}'
                })
            
            try:
                # Get sample ID
                sample_id = sample.get('__key__', f'sample_{processed_count}')
                
                # Get image data
                image_data = sample.get('image') or sample.get('jpg')
                if image_data is None:
                    failed_images.append({'sample_id': sample_id, 'reason': 'No image data'})
                    continue
                
                # Process image
                image_bytes = self._compress_image_fast(image_data, sample_id)
                if image_bytes is None:
                    continue
                
                # For train_small: we need another way to identify them
                # Option 1: Use the image itself (if it has metadata)
                # Option 2: Random sampling
                # Option 3: Use the first N samples
                
                if self.split == "train_small":
                    # Simple approach: take first 953K valid samples as train_small
                    # This is not ideal but works around the ID mismatch issue
                    if len(samples) >= 953_202:
                        break
                
                # Create a temporary label (will be refined later)
                taxonomic_label = f"species_{len(samples) % 1000}"  # Placeholder
                
                # Create sample
                samples.append({
                    'index': len(samples),
                    'taxonomic_label': taxonomic_label,
                    'treeoflife_id': sample_id,
                    'image_bytes': image_bytes,
                    'image_path': None,
                    'metadata': {'split': self.split}  # Minimal metadata
                })
                
                # Update progress
                pbar.update(1)
                
            except Exception as e:
                if processed_count <= 10:
                    print(f"\n⚠️ Error processing sample: {type(e).__name__}")
                continue
        
        pbar.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        
        print("\n⚠️  Warning: Using placeholder labels due to catalog mismatch")
        print("💡 For proper taxonomic labels, consider:")
        print("  1. Using local files with correct naming")
        print("  2. Finding a mapping between HF IDs and catalog IDs")
        
        return samples
    
    def _compress_image_fast(self, image: Any, sample_id: str) -> Optional[bytes]:
        """Fast image compression with minimal overhead."""
        try:
            # Handle PIL Image
            if isinstance(image, Image.Image):
                # Quick size check
                if image.width * image.height > 100_000_000:
                    return None
                
                # Convert mode if needed
                if image.mode not in ['RGB', 'L']:
                    image = image.convert('RGB')
                
                # Fast compression
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                return buffer.getvalue()
            
            # Handle bytes
            elif hasattr(image, 'read'):
                image_bytes = image.read()
                # Quick validation
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    if img.width * img.height > 100_000_000:
                        return None
                    # Return original bytes if valid
                    return image_bytes
                except:
                    return None
            
            return None
            
        except Exception:
            # Silent fail for speed
            return None
    
    def _should_log_error(self, error_type: str) -> bool:
        """Determine if we should log an error (to avoid spamming the console)."""
        if not hasattr(self, '_error_counts'):
            self._error_counts = {}
        
        count = self._error_counts.get(error_type, 0)
        self._error_counts[error_type] = count + 1
        
        # Only log first 5 errors of each type
        return count < 5
    
    def _get_taxonomy_from_catalog_optimized(self, sample_id: str) -> Optional[Dict[str, str]]:
        """OPTIMIZED: Ultra-fast taxonomy lookup using memory catalog."""
        # OPTIMIZATION 1: Use memory catalog if available (for train_small/val)
        if self.memory_catalog and sample_id in self.memory_catalog:
            return self.memory_catalog[sample_id]
        
        # OPTIMIZATION 2: Use in-memory cache for other splits
        if sample_id in self.catalog_cache:
            return self.catalog_cache[sample_id]
        
        # For train split, we shouldn't need to fall back to slow lookup
        # if memory catalog is properly loaded
        return None
    
    def _get_taxonomy_from_catalog_original(self, sample_id: str) -> Optional[Dict[str, str]]:
        """Original catalog lookup method as fallback using text index."""
        if not hasattr(self, 'index_path') or not os.path.exists(self.index_path):
            # Fallback to pandas if no index
            try:
                import pandas as pd
                # Read catalog if not already in memory
                if not hasattr(self, '_catalog_df'):
                    self._catalog_df = pd.read_csv(self.catalog_path)
                    self._catalog_df.set_index('treeoflife_id', inplace=True)
                
                if sample_id in self._catalog_df.index:
                    row = self._catalog_df.loc[sample_id]
                    return {
                        'kingdom': str(row.get('kingdom', '')).strip() if pd.notna(row.get('kingdom')) else '',
                        'phylum': str(row.get('phylum', '')).strip() if pd.notna(row.get('phylum')) else '',
                        'class': str(row.get('class', '')).strip() if pd.notna(row.get('class')) else '',
                        'order': str(row.get('order', '')).strip() if pd.notna(row.get('order')) else '',
                        'family': str(row.get('family', '')).strip() if pd.notna(row.get('family')) else '',
                        'genus': str(row.get('genus', '')).strip() if pd.notna(row.get('genus')) else '',
                        'species': str(row.get('species', '')).strip() if pd.notna(row.get('species')) else '',
                        'common': str(row.get('common', '')).strip() if pd.notna(row.get('common')) else '',
                        'split': str(row.get('split', 'train'))
                    }
                return None
            except Exception as e:
                return None
        
        try:
            # Search index file for line number
            target_line = None
            with open(self.index_path, 'r') as index_file:
                for line in index_file:
                    if line.startswith(f"{sample_id}:"):
                        target_line = int(line.split(':')[1].strip())
                        break
            
            if target_line is None:
                return None
            
            # Read specific line from catalog
            with open(self.catalog_path, 'r', encoding='utf-8') as csv_file:
                for current_line_num, line in enumerate(csv_file, 1):
                    if current_line_num == target_line:
                        # Parse the line manually
                        parts = self._parse_csv_line(line)
                        
                        # Map to taxonomy fields (based on catalog structure)
                        # Order: split,treeoflife_id,eol_content_id,eol_page_id,bioscan_part,bioscan_filename,
                        #        inat21_filename,inat21_cls_name,inat21_cls_num,kingdom,phylum,class,order,family,genus,species,common
                        return {
                            'kingdom': parts[9] if len(parts) > 9 else '',
                            'phylum': parts[10] if len(parts) > 10 else '',
                            'class': parts[11] if len(parts) > 11 else '',
                            'order': parts[12] if len(parts) > 12 else '',
                            'family': parts[13] if len(parts) > 13 else '',
                            'genus': parts[14] if len(parts) > 14 else '',
                            'species': parts[15] if len(parts) > 15 else '',
                            'common': parts[16] if len(parts) > 16 else '',
                            'split': parts[0] if len(parts) > 0 else 'train'
                        }
                        
            return None
            
        except Exception as e:
            return None
    
    def _is_valid_label(self, label: str) -> bool:
        """Check if a label is valid based on strict filtering rules."""
        if not label or not isinstance(label, str):
            return False
        
        label = label.strip().lower()
        
        # Basic invalid labels
        basic_invalid = ['', 'nan', 'none', 'n/a', 'null']
        if label in basic_invalid:
            return False
        
        if not self.strict_label_filtering:
            return True
        
        # Strict filtering: exclude uncertain/confusing labels
        uncertain_indicators = [
            'unknown', 'sp.', 'confusor', 'confuse', 'uncertain', 'unclear', 'ambiguous',
            'unidentified', 'unidentifiable', 'indeterminate', 'questionable'
        ]
        
        if label in uncertain_indicators:
            return False
        
        # Filter hybrid indicators
        if 'x ' in label or ' x ' in label:
            return False
        
        # Filter uncertain species indicators
        if 'sp.' in label or 'spp.' in label:
            return False
        
        # Filter cf., aff. (uncertain ID)
        if 'cf.' in label or 'aff.' in label:
            return False
        
        # Filter complex/group species
        if 'complex' in label or 'group' in label:
            return False
        
        # Filter near indicators
        if 'nr.' in label or 'near' in label:
            return False
        
        # Filter subspecies uncertainty
        if label.endswith(' ssp.') or ' ssp. ' in label:
            return False
        
        # Species-specific filters
        if self.taxonomic_level == 'species':
            # Filter names with numbers
            if any(char.isdigit() for char in label):
                return False
            
            # Filter very short species names
            parts = label.split()
            if len(parts) < 2:
                return False
            
            # Filter if parts contain uncertain indicators
            for part in parts:
                if any(indicator in part for indicator in ['sp', 'cf', 'aff', 'nr']):
                    return False
        
        return True

    def _get_taxonomic_label_from_dict(self, taxonomy: Dict[str, str]) -> Optional[str]:
        """Extract taxonomic label at the specified level with optional strict filtering."""
        if self.taxonomic_level not in taxonomy:
            return None
        
        label = str(taxonomy.get(self.taxonomic_level, '')).strip()
        
        if not self._is_valid_label(label):
            return None
        
        return label
    
    def _has_full_taxonomy_from_dict(self, taxonomy: Dict[str, str]) -> bool:
        """Check if taxonomy dict has labels for all major taxonomic levels."""
        required_fields = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for field in required_fields:
            value = str(taxonomy.get(field, '')).strip()
            if not value or value.lower() in ['nan', 'none', 'n/a', '']:
                return False
        
        return True
    
    def _has_full_taxonomy(self, row) -> bool:
        """Check if a pandas row has labels for all major taxonomic levels."""
        import pandas as pd
        required_fields = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for field in required_fields:
            if field not in row:
                return False
            value = row[field]
            if pd.isna(value) or str(value).strip() == '':
                return False
        
        return True
    
    def _filter_samples_by_class_count(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out classes with fewer than min_images_per_class samples."""
        if self.min_images_per_class <= 1:
            return samples

        print(f"\n🔍 Filtering classes with less than {self.min_images_per_class} images...")
        
        # Count occurrences of each class
        class_counts = Counter(sample['taxonomic_label'] for sample in samples)
        
        # Identify classes to keep
        classes_to_keep = {
            label for label, count in class_counts.items() 
            if count >= self.min_images_per_class
        }
        
        original_count = len(samples)
        original_class_count = len(class_counts)
        
        # Filter the samples
        filtered_samples = [
            sample for sample in samples 
            if sample['taxonomic_label'] in classes_to_keep
        ]
        
        final_class_count = len(classes_to_keep)
        
        print(f"✅ Filtering complete.")
        print(f"  Original classes: {original_class_count:,}, Final classes: {final_class_count:,}")
        print(f"  Original samples: {original_count:,}, Final samples: {len(filtered_samples):,}")
        
        return filtered_samples
    
    def _get_classes(self) -> List[str]:
        """Get list of class names in the dataset."""
        if not self.data:
            return []
        
        # Extract unique taxonomic labels
        classes = list(set(item['taxonomic_label'] for item in self.data))
        classes.sort()
        
        return classes
    
    def get_templates(self) -> List[str]:
        """Get prompt templates for zero-shot classification.
        
        For TreeOfLife, we can use templates that work well with biological specimens.
        
        Returns:
            List of prompt templates with {} placeholder for class name
        """
        # Import templates from the central registry
        from dataset_adapters import DatasetTemplates
        
        # Use the comprehensive template set from DatasetTemplates
        # These templates are designed to work well across different types of images
        # including natural specimens, which is perfect for TreeOfLife
        return DatasetTemplates.get_templates()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample, handling both file paths and byte data.
        
        Returns:
            image: Transformed image tensor
            label: Integer class label
        """
        try:
            item = self.data[idx]
            
            # Handle different image storage methods
            if 'image_path' in item and item['image_path']:
                # Load from file path
                image = Image.open(item['image_path']).convert('RGB')
            elif 'image_bytes' in item and item['image_bytes']:
                # Load from bytes (HuggingFace streaming mode)
                image = self._load_image_from_bytes(item['image_bytes'], item.get('treeoflife_id', f'sample_{idx}'))
                if image is None:
                    # Create placeholder image if loading fails
                    image = Image.new('RGB', (224, 224), (128, 128, 128))
            else:
                raise ValueError(f"No valid image source found for sample {idx}")
            
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    if self._should_log_error('transform'):
                        print(f"⚠️ Error applying transform at index {idx}: {type(e).__name__}")
                    # Create placeholder tensor
                    image = torch.zeros(3, 224, 224)
            
            # Convert label to index
            if isinstance(item['taxonomic_label'], str):
                label = self.class_to_idx[item['taxonomic_label']]
            else:
                label = item['taxonomic_label']
            
            return image, label
            
        except Exception as e:
            if self._should_log_error('getitem'):
                print(f"⚠️ Critical error in __getitem__ at index {idx}: {type(e).__name__}: {str(e)[:100]}")
            # Return placeholder data
            placeholder_image = torch.zeros(3, 224, 224)
            placeholder_label = 0
            return placeholder_image, placeholder_label
    
    def _load_image_from_bytes(self, image_bytes: bytes, sample_id: str) -> Optional[Image.Image]:
        """Load image from compressed bytes with robust error handling."""
        try:
            # Set a temporary warning filter for this specific operation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                image = Image.open(io.BytesIO(image_bytes))
                
                # Basic integrity check
                _ = image.size
                _ = image.mode
                
                # Convert problematic modes to RGB
                if image.mode not in ['RGB', 'L']:
                    image = image.convert('RGB')
                
                return image
                
        except (UnidentifiedImageError, OSError, IOError) as e:
            # Handle specific image format errors including TIFF errors
            error_msg = str(e).lower()
            if 'tiff' in error_msg or 'tif' in error_msg:
                # Silently skip TIFF errors - they're common in TreeOfLife
                pass
            elif self._should_log_error('decode'):
                print(f"⚠️ Failed to decode image {sample_id}: {str(e)[:50]}")
            return None
        except Exception as e:
            if self._should_log_error('unexpected_decode'):
                print(f"⚠️ Unexpected error decoding image {sample_id}: {type(e).__name__}")
            return None