"""TreeOfLife-10M dataset adapter with HuggingFace integration for efficient loading."""

import os
import sys
import time
import random
import sqlite3
import warnings
import io
import traceback
import pickle
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
        """Load dataset using either HuggingFace datasets or pandas+local images."""
        print(f"🌳 Loading TreeOfLife-10M data...")
        print(f"    Split: {self.split}")
        print(f"    Streaming: {self.streaming}")
        print(f"    Max samples: {self.max_samples}")
        print(f"    Taxonomic level: {self.taxonomic_level}")
        print(f"    Smart sampling: {self.use_smart_sampling}")
        
        self.catalog_path = self._ensure_catalog_csv()
        
        # OPTIMIZATION: Pre-load catalog into memory for ultra-fast lookups
        if self.split in ["train_small", "val"] or self.use_smart_sampling:
            print("🚀 OPTIMIZATION: Pre-loading catalog for ultra-fast lookups...")
            self._preload_catalog_for_split()
        
        # Determine loading method based on streaming parameter
        if self.streaming:
            print("📡 Using HuggingFace online streaming mode")
            return self._load_huggingface_data_optimized()
        else:
            # Check for local images when not streaming
            if self._check_local_images():
                print("💾 Using local files mode")
                return self._load_pandas_data()
            else:
                print("⚠️ Streaming is False but no local images found. Falling back to streaming mode.")
                return self._load_huggingface_data_optimized()
    
    def _check_local_images(self):
        """Check if local images exist for the current split."""
        try:
            # For train_small, check if any jpg files exist
            if self.split == "train_small":
                jpg_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')]
                if jpg_files:
                    print(f"✅ Found {len(jpg_files)} local images in {self.images_dir}")
                    return True
        except Exception:
            pass
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
            df = pd.read_csv(self.catalog_path, usecols=usecols)
            
            print(f"📊 Catalog loaded: {len(df):,} total entries")
            
            # Filter by split
            df_split = df[df['split'] == self.split].copy()
            print(f"📊 Filtered to {self.split}: {len(df_split):,} entries")
            
            # Convert to dictionary for fast lookups
            for _, row in df_split.iterrows():
                sample_id = row['treeoflife_id']
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
            print(f"❌ Failed to pre-load catalog: {e}")
            # Fallback to original method
            self.memory_catalog = None
            self.split_sample_ids = None
    
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
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        print(f"🔗 Loading streaming dataset...")
        
        try:
            # HuggingFace dataset only provides 'train' split
            # We filter by catalog for train_small and val
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return []  # Return empty list on failure

        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        skipped_wrong_split = 0
        start_time = time.time()
        
        # For performance optimization: batch processing
        batch_size = 100
        batch_buffer = []

        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # Determine target number of samples for the progress bar
        if self.split_sample_ids:
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} {self.split} entries")
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            if self.max_samples:
                print(f"🎯 Target: {self.max_samples:,} valid samples")
            else:
                print(f"🎯 Processing to find {self.split} samples (~{target_samples:,} total to scan)")

        from tqdm import tqdm
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                if self.max_samples and len(samples) >= self.max_samples:
                    print(f"\n🏁 Reached max_samples limit of {self.max_samples}. Stopping.")
                    break

                # Add to batch buffer for processing
                batch_buffer.append(sample)
                
                # Process batch when buffer is full
                if len(batch_buffer) >= batch_size:
                    batch_results = self._process_sample_batch(
                        batch_buffer, failed_images, catalog_misses, skipped_wrong_split
                    )
                    samples.extend(batch_results['samples'])
                    catalog_misses = batch_results['catalog_misses']
                    skipped_wrong_split = batch_results['skipped_wrong_split']
                    
                    # Update progress
                    if batch_results['samples']:
                        pbar.update(len(batch_results['samples']))
                        pbar.set_postfix(
                            found=len(samples), 
                            failed=len(failed_images),
                            wrong_split=skipped_wrong_split,
                            refresh=True
                        )
                    
                    batch_buffer = []
                    processed_count += batch_size
            
            # Process remaining samples in buffer
            if batch_buffer:
                batch_results = self._process_sample_batch(
                    batch_buffer, failed_images, catalog_misses, skipped_wrong_split
                )
                samples.extend(batch_results['samples'])
                catalog_misses = batch_results['catalog_misses']
                skipped_wrong_split = batch_results['skipped_wrong_split']
                processed_count += len(batch_buffer)
                if batch_results['samples']:
                    pbar.update(len(batch_results['samples']))
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading/processing failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        print(f"  🔀 Skipped wrong split: {skipped_wrong_split:,}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            return []
        
        print(f"\n✅ Successfully loaded {len(samples):,} valid samples from TreeOfLife-10M")
        return self._filter_samples_by_class_count(samples)
    
    def _process_sample_batch(self, batch: List[Dict], failed_images: List[Dict], 
                            catalog_misses: int, skipped_wrong_split: int) -> Dict:
        """Process a batch of samples for better performance."""
        batch_samples = []
        
        for sample in batch:
            try:
                if not isinstance(sample, dict):
                    failed_images.append({'sample_id': 'N/A', 'reason': f'Invalid sample format: {type(sample)}'})
                    continue

                # Try different field names for sample_id
                sample_id = sample.get('treeoflife_id') or sample.get('__key__')
                if not sample_id:
                    failed_images.append({'sample_id': 'Unknown', 'reason': 'Missing treeoflife_id'})
                    continue

                # Get metadata from catalog
                metadata = self._get_taxonomy_from_catalog_optimized(sample_id)
                if metadata is None:
                    catalog_misses += 1
                    continue

                # Check if it's the correct split
                sample_split = metadata.get('split', 'train')
                if self.split == 'train' and sample_split == 'val':
                    # Skip val when loading train (but include train_small in train)
                    skipped_wrong_split += 1
                    continue
                elif self.split in ['train_small', 'val'] and sample_split != self.split:
                    # For train_small and val, only load exact matches
                    skipped_wrong_split += 1
                    continue

                # Get label at specified taxonomic level
                label = self._get_taxonomic_label_from_dict(metadata)
                if not label:
                    continue

                # Skip if excluding partial labels
                if self.exclude_partial_labels and not self._has_full_taxonomy_from_dict(metadata):
                    continue

                # Try different field names for image data
                image_data = sample.get('image') or sample.get('jpg')
                if image_data is None:
                    failed_images.append({'sample_id': sample_id, 'reason': 'Missing image data'})
                    continue

                # Compress image - this handles TIFF errors gracefully
                image_bytes = self._compress_image(image_data, sample_id)
                if image_bytes is None:
                    # Error already logged in _compress_image, just continue
                    continue
                
                # Create processed sample
                processed_sample = {
                    "index": len(batch_samples),
                    "taxonomic_label": label,
                    "treeoflife_id": sample_id,
                    "image_bytes": image_bytes,
                    "metadata": metadata
                }
                batch_samples.append(processed_sample)

            except Exception as e:
                # Catch any unexpected errors and continue processing
                sample_id_for_error = sample.get('treeoflife_id', sample.get('__key__', 'Unknown'))
                print(f"\n⚠️ Unexpected error processing sample {sample_id_for_error}: {e}")
                failed_images.append({'sample_id': sample_id_for_error, 'reason': str(e)})
                continue
        
        return {
            'samples': batch_samples,
            'catalog_misses': catalog_misses,
            'skipped_wrong_split': skipped_wrong_split
        }
    
    def _compress_image(self, image: Any, sample_id: str) -> Optional[bytes]:
        """
        Compresses a PIL image or image bytes into JPEG format.
        Handles corrupt images and logs errors.
        """
        try:
            if not isinstance(image, Image.Image):
                # Assuming it's bytes, try to open it
                if hasattr(image, 'read'):
                    image_bytes = image.read()
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # If it's not a file-like object, we can't process it
                    print(f"⚠️ Skipping sample {sample_id}: Invalid image type {type(image)}")
                    return None

            # Check for excessively large images before processing
            width, height = image.size
            if width * height > 100_000_000:  # 100 megapixels limit
                print(f"⚠️ Skipping sample {sample_id}: Image too large ({width}x{height})")
                return None

            # Convert to RGB if necessary
            if image.mode in ['RGBA', 'LA', 'P']:
                image = image.convert('RGB')
            
            # Save to in-memory bytes buffer
            bytes_io = io.BytesIO()
            image.save(bytes_io, format='JPEG', quality=85, optimize=True)
            return bytes_io.getvalue()

        except (UnidentifiedImageError, OSError, IOError) as e:
            # More specific error logging for corrupt images
            if "not a TIFF file" in str(e):
                print(f"⚠️ Corrupt TIFF image skipped (ID: {sample_id}). Error: {e}")
            else:
                print(f"⚠️ Corrupt image skipped (ID: {sample_id}). Error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error compressing image for sample {sample_id}: {e}")
            return None

    def _get_taxonomy_from_catalog_optimized(self, sample_id: str) -> Optional[Dict[str, str]]:
        """OPTIMIZED: Ultra-fast taxonomy lookup using memory catalog."""
        # OPTIMIZATION 1: Use memory catalog if available
        if self.memory_catalog and sample_id in self.memory_catalog:
            return self.memory_catalog[sample_id]
        
        # OPTIMIZATION 2: Use in-memory cache
        if sample_id in self.catalog_cache:
            return self.catalog_cache[sample_id]
        
        # Fallback to reading catalog
        taxonomy = self._get_taxonomy_from_catalog_original(sample_id)
        
        # Add to cache
        if taxonomy and len(self.catalog_cache) < self.catalog_cache_size:
            self.catalog_cache[sample_id] = taxonomy
        elif taxonomy and len(self.catalog_cache) >= self.catalog_cache_size:
            # Remove 25% of cache when full
            keys_to_remove = list(self.catalog_cache.keys())[:self.catalog_cache_size//4]
            for key in keys_to_remove:
                del self.catalog_cache[key]
            self.catalog_cache[sample_id] = taxonomy
        
        return taxonomy
    
    def _get_taxonomy_from_catalog_original(self, sample_id: str) -> Optional[Dict[str, str]]:
        """Original catalog lookup method as fallback."""
        try:
            import pandas as pd
            # Read catalog if not already in memory
            if not hasattr(self, '_catalog_df'):
                self._catalog_df = pd.read_csv(self.catalog_path)
                self._catalog_df.set_index('treeoflife_id', inplace=True)
            
            if sample_id in self._catalog_df.index:
                row = self._catalog_df.loc[sample_id]
                return {
                    'kingdom': str(row.get('kingdom', '')) if pd.notna(row.get('kingdom')) else '',
                    'phylum': str(row.get('phylum', '')) if pd.notna(row.get('phylum')) else '',
                    'class': str(row.get('class', '')) if pd.notna(row.get('class')) else '',
                    'order': str(row.get('order', '')) if pd.notna(row.get('order')) else '',
                    'family': str(row.get('family', '')) if pd.notna(row.get('family')) else '',
                    'genus': str(row.get('genus', '')) if pd.notna(row.get('genus')) else '',
                    'species': str(row.get('species', '')) if pd.notna(row.get('species')) else '',
                    'common': str(row.get('common', '')) if pd.notna(row.get('common')) else '',
                    'split': str(row.get('split', 'train'))
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