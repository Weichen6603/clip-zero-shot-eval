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
    
    Features:
    - HuggingFace streaming for efficient data loading
    - Pandas+本地图片+catalog.csv 方式支持 train_small 快速实验
    - Flexible taxonomic level selection (species, genus, family, etc.)
    - **SPEED OPTIMIZATIONS:**
      - Memory-cached catalog for ultra-fast lookups
      - Batch processing for reduced overhead
      - Smart sampling strategies
    """
    
    def __init__(self, root_path: str = "./data/treeoflife", transform=None, split: str = "train", 
                 max_samples: Optional[int] = None, taxonomic_level: str = "species",
                 min_images_per_class: int = 1, exclude_partial_labels: bool = False,
                 max_shards: Optional[int] = None, strict_label_filtering: bool = False, 
                 use_pandas: bool = False, images_dir: Optional[str] = None, 
                 catalog_cache_size: int = 50000, use_smart_sampling: bool = True, **kwargs):
        """
        Initialize TreeOfLife-10M adapter.
        
        Args:
            root_path: Path to dataset cache directory (default: ./data/treeoflife)
            transform: Image transformations to apply
            split: Dataset split ('train', 'train_small')
            max_samples: Maximum number of samples to use (None for all)
            taxonomic_level: Level of taxonomy to use for classification 
                           ('species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom')
            min_images_per_class: Minimum images per class to include class
            exclude_partial_labels: Whether to exclude images without full taxonomic labels
            max_shards: Maximum number of data shards to download/use (None for all)
            strict_label_filtering: Whether to apply strict filtering of uncertain/confusing labels (default: False)
            use_pandas: 是否用 Pandas+本地图片方式加载 train_small
            images_dir: 本地图片解压目录（如 /path/to/images）
            catalog_cache_size: Size of in-memory catalog cache (default: 50000)
            use_smart_sampling: Whether to use smart sampling for faster train_small loading
        """
        self.max_samples = max_samples
        self.taxonomic_level = taxonomic_level.lower()
        self.min_images_per_class = min_images_per_class
        self.exclude_partial_labels = exclude_partial_labels
        self.max_shards = max_shards
        self.strict_label_filtering = strict_label_filtering
        self.catalog_db_path = None  # Will hold SQLite database path
        self.use_pandas = use_pandas
        self.images_dir = images_dir or os.path.join(root_path, "images")
        self.split = split
        self.root_path = root_path
        self.catalog_cache = {}  # Initialize cache for catalog lookups
        self.catalog_cache_size = catalog_cache_size  # Increased cache size
        self.use_smart_sampling = use_smart_sampling
        
        # New optimization attributes
        self.memory_catalog = None  # Full in-memory catalog for train_small
        self.split_sample_ids = None  # Pre-filtered sample IDs for target split
        
        # Validate taxonomic level
        valid_levels = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
        if self.taxonomic_level not in valid_levels:
            raise ValueError(f"taxonomic_level must be one of {valid_levels}")
        
        # Call parent constructor with required parameters
        super().__init__(root_path=root_path, transform=transform, split=split, **kwargs)
        
    def _ensure_catalog_csv(self):
        """Ensure catalog.csv exists locally. Download from HuggingFace if missing, and return the local metadata path."""
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
        print(f"    Use Pandas: {self.use_pandas}")
        print(f"    Max samples: {self.max_samples}")
        print(f"    Max shards: {self.max_shards}")
        print(f"    Smart sampling: {self.use_smart_sampling}")
        
        self.catalog_path = self._ensure_catalog_csv()
        
        # OPTIMIZATION: For train_small, pre-load catalog into memory for ultra-fast lookups
        if self.split == "train_small" or self.use_smart_sampling:
            print("🚀 OPTIMIZATION: Pre-loading catalog for ultra-fast lookups...")
            self._preload_catalog_for_split()
        
        # Handle train_small subset
        if self.split == "train_small":
            # Check for local images to use Pandas loader
            image_dir = self.images_dir
            jpg_files = []
            try:
                jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
            except Exception:
                jpg_files = []
            if jpg_files:
                print(f"ℹ️ Local images detected ({len(jpg_files)} files). Using Pandas loader for 'train_small'.")
                return self._load_pandas_train_small_data()
            else:
                print("ℹ️ No local images found. Using optimized HuggingFace streaming for 'train_small'.")
                return self._load_huggingface_data_optimized()
        # Other modes
        if self.use_pandas:
            print("ℹ️ Using Pandas mode for custom filtering.")
            return self._load_pandas_train_small_data()
        else:
            return self._load_huggingface_data_optimized()
    
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
        self.split_sample_ids = []
        
        # Read catalog with pandas for speed
        try:
            import pandas as pd
            print("📊 Reading catalog with pandas...")
            
            # Read only necessary columns to save memory
            usecols = ['split', 'treeoflife_id', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'common']
            df = pd.read_csv(self.catalog_path, usecols=usecols)
            
            print(f"📊 Catalog loaded: {len(df):,} total entries")
            
            # Filter by split if needed
            if self.split == "train_small":
                df_split = df[df['split'] == 'train_small'].copy()
                print(f"📊 Filtered to {self.split}: {len(df_split):,} entries")
            else:
                df_split = df.copy()
            
            # Convert to dictionary for fast lookups
            for _, row in df_split.iterrows():
                sample_id = row['treeoflife_id']
                self.memory_catalog[sample_id] = {
                    'kingdom': str(row.get('kingdom', '')).strip(),
                    'phylum': str(row.get('phylum', '')).strip(),
                    'class': str(row.get('class', '')).strip(),
                    'order': str(row.get('order', '')).strip(),
                    'family': str(row.get('family', '')).strip(),
                    'genus': str(row.get('genus', '')).strip(),
                    'species': str(row.get('species', '')).strip(),
                    'common': str(row.get('common', '')).strip(),
                    'split': str(row.get('split', 'train')).strip()
                }
                self.split_sample_ids.append(sample_id)
            
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
    
    def _load_pandas_train_small_data(self) -> List[Dict[str, Any]]:
        """Load train_small subset using Pandas and local images, with catalog.csv filtering."""
        import pandas as pd
        from tqdm import tqdm
        catalog_path = self.catalog_path  # unified entry
        print(f"📋 Reading catalog.csv: {catalog_path}")
        df = pd.read_csv(catalog_path)
        df_small = df[df["split"] == "train_small"].copy()
        print(f"✅ train_small sample count: {len(df_small)}")
        if self.max_samples:
            df_small = df_small.sample(n=min(self.max_samples, len(df_small)), random_state=42)
        # Build image path
        def get_image_path(treeoflife_id):
            return os.path.join(self.images_dir, f"{treeoflife_id}.jpg")
        df_small["image_path"] = df_small["treeoflife_id"].apply(get_image_path)
        # Select label
        label_col = self.taxonomic_level if self.taxonomic_level in df_small.columns else "species"
        df_small["label"] = df_small[label_col]
        # Filter invalid labels
        df_small = df_small[df_small["label"].notnull() & (df_small["label"].astype(str).str.strip() != "")]
        # Build samples
        samples = []
        for idx, row in tqdm(df_small.iterrows(), total=len(df_small), desc="Building samples"):
            image_path = row["image_path"]
            if not os.path.exists(image_path):
                continue  # skip missing images
            samples.append({
                "index": idx,
                "taxonomic_label": row["label"],
                "treeoflife_id": row["treeoflife_id"],
                "image_path": image_path,
                "metadata": row.to_dict(),
            })
        print(f"✅ Valid samples: {len(samples)}")
        return self._filter_samples_by_class_count(samples)
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        split_to_load = "train"
        print(f"🔗 Loading streaming dataset (split: {split_to_load})...")
        dataset = load_dataset(
            dataset_name,
            split=split_to_load,
            streaming=True,
            cache_dir=self.root_path
        )
        
        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        start_time = time.time()
        
        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # OPTIMIZATION: Smart sampling strategy
        estimated_total = self._estimate_total_samples()
        
        # For train_small with memory catalog, use smart sampling
        if self.split == "train_small" and self.memory_catalog and self.split_sample_ids:
            print("🚀 SMART SAMPLING: Using pre-filtered sample IDs for ultra-fast processing")
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} train_small entries")
        else:
            # Standard processing
            if self.max_samples:
                scan_limit = min(self.max_samples * 3, estimated_total)
                print(f"🎯 Target: {self.max_samples} valid samples (scanning up to {scan_limit:,} total)")
                estimated_total = scan_limit
            elif self.max_shards:
                shard_estimate = self.max_shards * 140000
                estimated_total = min(estimated_total, shard_estimate)
                print(f"🎯 Processing {self.max_shards} shards (~{estimated_total:,} samples)")
            else:
                print(f"🎯 Processing full dataset (~{estimated_total:,} samples)")
        
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=estimated_total,
                desc="Processing samples",
                unit="samples",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True
            )
        except ImportError:
            progress_bar = None
            print("💡 Install tqdm for better progress tracking: pip install tqdm")
        
        # OPTIMIZATION: Batch processing for reduced overhead
        batch_size = 100
        sample_batch = []
        
        # Main loop for processing samples
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                try:
                    # OPTIMIZATION: Use pre-filtered sample IDs for ultra-fast filtering
                    if self.split_sample_ids and sample['treeoflife_id'] not in self.split_sample_ids:
                        pbar.update(1)
                        continue

                    # Get metadata from memory catalog if available
                    metadata = self.memory_catalog.get(sample['treeoflife_id'])
                    if metadata is None:
                        catalog_misses += 1
                        continue  # Skip samples not in catalog

                    # Extract taxonomic label
                    label = metadata.get(self.taxonomic_level)
                    if not label:
                        continue

                    # Store compressed image bytes to save memory
                    image_bytes = self._compress_image(sample['image'])
                    if image_bytes is None:
                        failed_images.append(sample['treeoflife_id'])
                        continue
                    
                    processed_sample = {
                        "index": processed_count,
                        "taxonomic_label": label,
                        "treeoflife_id": sample['treeoflife_id'],
                        "image_bytes": image_bytes,  # Store compressed bytes
                        "metadata": metadata
                    }
                    samples.append(processed_sample)
                    pbar.set_postfix(found=len(samples), failed=len(failed_images), refresh=True)

                except (UnidentifiedImageError, OSError, IOError) as e:
                    sample_id = sample.get('treeoflife_id', 'Unknown ID')
                    print(f"\n⚠️ Corrupt image skipped during initial loading (ID: {sample_id}). Error: {e}")
                    failed_images.append(sample_id)
                    continue # Skip to the next sample
                except Exception as e:
                    sample_id = sample.get('treeoflife_id', 'Unknown ID')
                    print(f"\nUnexpected error processing sample ID {sample_id}: {e}")
                    traceback.print_exc()
                    failed_images.append(sample_id)
                    continue


                processed_count += 1
                pbar.update(1)
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Display loading statistics
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            raise ValueError("No valid samples extracted from TreeOfLife-10M dataset")
        
        print(f"\n✅ Successfully loaded {len(samples)} valid samples from TreeOfLife-10M")
        return self._filter_samples_by_class_count(samples)
    
    def _process_sample_batch(self, sample_batch: List[Tuple[int, Dict]], failed_images: List[Dict], catalog_misses: int) -> List[Dict[str, Any]]:
        """OPTIMIZATION: Process samples in batches for better performance."""
        valid_samples = []
        
        for idx, sample in sample_batch:
            try:
                sample_info = self._process_hf_sample_optimized(sample, idx, failed_images)
                if sample_info:
                    # If filtering by split, skip mismatched
                    if self.split == "train_small":
                        if sample_info.get('metadata', {}).get('split') != self.split:
                            continue
                    valid_samples.append(sample_info)
                else:
                    # Check if it was a catalog miss
                    sample_id = sample.get('__key__', f'sample_{idx}')
                    if not self._get_taxonomy_from_catalog_optimized(sample_id):
                        catalog_misses += 1
                        
            except Exception as e:
                if idx < 10:  # Show details for first few errors
                    print(f"⚠️ Critical error processing sample {idx}: {e}")
                continue
        
        return valid_samples
    
    def _process_hf_sample_optimized(self, sample: Dict[str, Any], idx: int, failed_images: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """OPTIMIZED: Process a single sample with optimized catalog lookup."""
        try:
            sample_id = sample.get('__key__', f'sample_{idx}')
            sample_url = sample.get('__url__', '')
            
            # For train_small with smart sampling, skip non-target samples early
            if (self.split == "train_small" and self.split_sample_ids and 
                sample_id not in self.split_sample_ids):
                return None
            
            # Extract image from TreeOfLife-10M format
            image = sample.get('jpg')
            if image is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None
            
            # Process and store only compressed image bytes for true streaming
            image_bytes = None
            try:
                if not isinstance(image, Image.Image):
                    if hasattr(image, 'read'):
                        image_bytes = image.read()
                        try:
                            test_image = Image.open(io.BytesIO(image_bytes))
                            _ = test_image.size
                            
                            width, height = test_image.size
                            if width * height > 100_000_000:
                                failed_images.append({
                                    'sample_id': sample_id,
                                    'reason': f'Image too large: {width}x{height} pixels',
                                    'url': sample_url
                                })
                                return None
                            
                            del test_image
                        except (UnidentifiedImageError, OSError, IOError) as e:
                            sample_id = item.get('treeoflife_id', f'index_{idx}')
                            # More specific error logging
                            if "not a TIFF file" in str(e):
                                print(f"⚠️ Corrupt TIFF image skipped (ID: {sample_id}). Error: {e}")
                            else:
                                print(f"⚠️ Corrupt image skipped (ID: {sample_id}): {e}")
                            return None
                    else:
                        failed_images.append({
                            'sample_id': sample_id,
                            'reason': f'Invalid image type: {type(image)}',
                            'url': sample_url
                        })
                        return None
                else:
                    width, height = image.size
                    if width * height > 100_000_000:
                        failed_images.append({
                            'sample_id': sample_id,
                            'reason': f'Image too large: {width}x{height} pixels',
                            'url': sample_url
                        })
                        return None
                    
                    bytes_io = io.BytesIO()
                    if image.mode in ['RGBA', 'LA', 'P']:
                        image = image.convert('RGB')
                    elif image.mode not in ['RGB', 'L']:
                        image = image.convert('RGB')
                    
                    image.save(bytes_io, format='JPEG', quality=85, optimize=True)
                    image_bytes = bytes_io.getvalue()
                    del bytes_io
                
                del image
                
            except Exception as img_error:
                error_type = type(img_error).__name__
                error_msg = str(img_error)
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': f'{error_type}: {error_msg}',
                    'url': sample_url
                })
                return None
            
            # OPTIMIZED: Use memory catalog for ultra-fast lookup
            taxonomy = self._get_taxonomy_from_catalog_optimized(sample_id)
            
            if not taxonomy:
                return None
            
            # Extract taxonomic label at specified level
            taxonomic_label = self._get_taxonomic_label_from_dict(taxonomy)
            if not taxonomic_label:
                return None
            
            # Skip if excluding partial labels and sample doesn't have full taxonomy
            if self.exclude_partial_labels and not self._has_full_taxonomy_from_dict(taxonomy):
                return None
                
            sample_info = {
                'index': idx,
                'taxonomic_label': taxonomic_label,
                'common_name': taxonomy.get('common', ''),
                'scientific_name': taxonomy.get('species', ''),
                'treeoflife_id': sample_id,
                'metadata': taxonomy,
                'image_source': {
                    'type': 'compressed_bytes',
                    'image_bytes': image_bytes,
                    'sample_id': sample_id,
                }
            }
            
            return sample_info
            
        except Exception as e:
            return None
    
    def _get_taxonomy_from_catalog_optimized(self, sample_id: str) -> Optional[Dict[str, str]]:
        """OPTIMIZED: Ultra-fast taxonomy lookup using memory catalog."""
        # OPTIMIZATION 1: Use memory catalog if available (for train_small)
        if self.memory_catalog and sample_id in self.memory_catalog:
            return self.memory_catalog[sample_id]
        
        # OPTIMIZATION 2: Use larger in-memory cache
        if sample_id in self.catalog_cache:
            return self.catalog_cache[sample_id]
        
        # Fallback to original method for samples not in memory catalog
        taxonomy = self._get_taxonomy_from_catalog_original(sample_id)
        
        # Add to cache with larger size
        if taxonomy and len(self.catalog_cache) < self.catalog_cache_size:
            self.catalog_cache[sample_id] = taxonomy
        elif taxonomy and len(self.catalog_cache) >= self.catalog_cache_size:
            # Remove 25% of cache when full (keep more entries)
            keys_to_remove = list(self.catalog_cache.keys())[:self.catalog_cache_size//4]
            for key in keys_to_remove:
                del self.catalog_cache[key]
            self.catalog_cache[sample_id] = taxonomy
        
        return taxonomy
    
    def _get_taxonomy_from_catalog_original(self, sample_id: str) -> Optional[Dict[str, str]]:
        """Original catalog lookup method as fallback."""
        if not hasattr(self, 'index_path') or not os.path.exists(self.index_path):
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
                        taxonomy = self._parse_catalog_line(line)
                        return taxonomy
                        
            return None
            
        except Exception as e:
            return None

    def _get_taxonomic_label_from_dict(self, taxonomy: Dict[str, str]) -> Optional[str]:
        """Extract taxonomic label at the specified level with optional strict filtering."""
        level_map = {
            'kingdom': 'kingdom',
            'phylum': 'phylum', 
            'class': 'class',
            'order': 'order',
            'family': 'family',
            'genus': 'genus',
            'species': 'species'
        }
        
        if self.taxonomic_level not in level_map:
            return None
        
        field_name = level_map[self.taxonomic_level]
        label = str(taxonomy.get(field_name, '')).strip()
        
        # Basic filtering: always filter empty or obviously invalid labels
        basic_invalid = ['', 'nan', 'none', 'n/a', 'null']
        if not label or label.lower() in basic_invalid:
            return None
        
        # Apply strict filtering only if enabled
        if self.strict_label_filtering:
            # Strict filtering: Return None for uncertain/confusing/hybrid labels
            uncertain_indicators = [
                'unknown', 'sp.', 'confusor', 'confuse', 'uncertain', 'unclear', 'ambiguous',
                'unidentified', 'unidentifiable', 'indeterminate', 'questionable'
            ]
            
            if label.lower() in uncertain_indicators:
                return None
            
            # Filter hybrid indicators (x species, hybrids)
            if 'x ' in label.lower() or ' x ' in label.lower():
                return None
            
            # Filter uncertain species indicators
            if 'sp.' in label or 'spp.' in label:
                return None
                
            # Filter specimens with cf. (compare with), aff. (affinity with) - uncertain ID
            if 'cf.' in label.lower() or 'aff.' in label.lower():
                return None
                
            # Filter complex species (species complex, species group)
            if 'complex' in label.lower() or 'group' in label.lower():
                return None
                
            # Filter near/close to (nr./near) indicators
            if 'nr.' in label.lower() or 'near' in label.lower():
                return None
                
            # Filter subspecies indicators that suggest uncertainty
            if label.endswith(' ssp.') or ' ssp. ' in label:
                return None
                
            # Additional filters for specific taxonomic levels
            if self.taxonomic_level == 'species':
                # For species level, be extra strict
                # Filter names with numbers (often indicate forms/varieties/uncertain status)
                if any(char.isdigit() for char in label):
                    return None
                    
                # Filter very short species names (likely incomplete)
                parts = label.split()
                if len(parts) < 2:  # Species should have at least genus + species
                    return None
                    
                # Filter if genus or species part contains uncertain indicators
                for part in parts:
                    if any(indicator in part.lower() for indicator in ['sp', 'cf', 'aff', 'nr']):
                        return None
        
        return label
    
    def _has_full_taxonomy_from_dict(self, taxonomy: Dict[str, str]) -> bool:
        """Check if taxonomy dict has labels for all major taxonomic levels."""
        required_fields = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for field in required_fields:
            value = str(taxonomy.get(field, '')).strip()
            if not value or value.lower() in ['nan', 'none', 'n/a']:
                return False
        
        return True
    
    def _filter_samples_by_class_count(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples to only include classes with minimum image count."""
        if not samples:
            return samples
            
        # Count samples per class
        class_counts = defaultdict(int)
        for sample in samples:
            class_counts[sample['taxonomic_label']] += 1
        
        # Filter classes by minimum count
        valid_classes = {cls for cls, count in class_counts.items() 
                        if count >= self.min_images_per_class}
        
        # Filter samples
        filtered_samples = [s for s in samples if s['taxonomic_label'] in valid_classes]
        
        print(f"📊 Class filtering: {len(class_counts)} total classes, {len(valid_classes)} classes with ≥{self.min_images_per_class} samples")
        print(f"📊 Sample filtering: {len(samples)} → {len(filtered_samples)} samples")
        
        if valid_classes:
            print(f"📊 Classes found: {sorted(list(valid_classes))}")
            print(f"📊 Class distribution: {dict(Counter(class_counts).most_common())}")
        
        return filtered_samples
    
    def _get_classes(self) -> List[str]:
        """Extract unique class names from the dataset."""
        if hasattr(self, '_cached_classes'):
            return self._cached_classes
        
        # Extract classes from loaded data
        classes = sorted(list(set(sample['taxonomic_label'] for sample in self.data)))
        self._cached_classes = classes
        return classes
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample by index, supporting pandas+local images方式."""
        try:
            sample_info = self.data[index]
            # Pandas+本地图片方式
            if self.use_pandas and "image_path" in sample_info:
                image = Image.open(sample_info["image_path"]).convert("RGB")
            else:
                image = self._load_image_on_demand(sample_info)
                if image is None:
                    image = Image.new('RGB', (224, 224), (128, 128, 128))
            # Apply transform if specified
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    if not hasattr(self, '_transform_errors'):
                        self._transform_errors = 0
                    self._transform_errors += 1
                    if self._transform_errors < 5:
                        print(f"⚠️ Error applying transform at index {index}: {type(e).__name__}")
                    image = torch.zeros(3, 224, 224)
            else:
                to_tensor = transforms.ToTensor()
                try:
                    image = to_tensor(image)
                except Exception as e:
                    image = torch.zeros(3, 224, 224)
            label = self.class_to_idx[sample_info['taxonomic_label']]
            return image, label
        except Exception as e:
            print(f"⚠️ Critical error in __getitem__ at index {index}: {type(e).__name__}: {str(e)}")
            placeholder_image = torch.zeros(3, 224, 224)
            placeholder_label = 0
            return placeholder_image, placeholder_label
    
    def _load_image_on_demand(self, sample_info: Dict[str, Any]) -> Optional[Image.Image]:
        """Load image on-demand from compressed bytes for true streaming behavior."""
        try:
            image_source = sample_info.get('image_source', {})
            if image_source.get('type') == 'compressed_bytes':
                # Extract compressed image bytes
                image_bytes = image_source.get('image_bytes')
                
                if image_bytes is None:
                    return None
                
                # Decode from compressed bytes with robust error handling
                try:
                    # Set a temporary warning filter for this specific operation
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        image = Image.open(io.BytesIO(image_bytes))
                except (UnidentifiedImageError, OSError, IOError, ValueError) as e:
                    # Handle specific image format errors including TIFF errors
                    error_msg = str(e).lower()
                    if 'tiff' in error_msg or 'tif' in error_msg:
                        if not hasattr(self, '_tiff_errors'):
                            self._tiff_errors = 0
                        self._tiff_errors += 1
                        if self._tiff_errors < 5:  # Only print first 5 TIFF errors
                            print(f"⚠️ Skipping TIFF format error: {e}")
                    else:
                        print(f"⚠️ Failed to decode image: {e}")
                    return None
                except Exception as e:
                    # Catch any other unexpected errors
                    print(f"⚠️ Unexpected error decoding image: {type(e).__name__}: {e}")
                    return None
                
                # Basic integrity checks
                try:
                    _ = image.size
                    _ = image.mode
                    
                    # Convert problematic modes to RGB
                    if image.mode not in ['RGB', 'L']:
                        if image.mode in ['RGBA', 'LA', 'P']:
                            image = image.convert('RGB')
                        else:
                            # For any other exotic modes
                            image = image.convert('RGB')
                        
                except Exception as e:
                    if not hasattr(self, '_integrity_errors'):
                        self._integrity_errors = 0
                    self._integrity_errors += 1
                    if self._integrity_errors < 5:
                        print(f"⚠️ Image integrity check failed: {e}")
                    return None
                
                # Resize very large images if needed
                try:
                    width, height = image.size
                    if width * height > 50_000_000:  # 50MP limit for runtime
                        max_size = 2048
                        if width > max_size or height > max_size:
                            ratio = min(max_size / width, max_size / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except Exception as e:
                    if not hasattr(self, '_resize_errors'):
                        self._resize_errors = 0
                    self._resize_errors += 1
                    if self._resize_errors < 5:
                        print(f"⚠️ Image resizing failed: {e}")
                    return None
                
                return image
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading image on-demand: {type(e).__name__}: {e}")
            return None
    
    def get_templates(self) -> List[str]:
        """Get text templates for TreeOfLife-10M dataset."""
        from dataset_adapters import DatasetTemplates
        # Use unified templates for consistency across all datasets
        return DatasetTemplates.get_templates()
    
    def _init_catalog_lazy_reader(self, catalog_path: str):
        """Initialize ultra-lightweight catalog reader using text-based indexing."""
        self.catalog_path = catalog_path
        
        # If we already have memory catalog, skip text index creation
        if self.memory_catalog:
            print("✅ Using memory catalog, skipping text index")
            return
            
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
                        parts = []
                        current_part = ""
                        in_quotes = False
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                parts.append(current_part.strip('"'))
                                current_part = ""
                                continue
                            current_part += char
                        
                        if current_part:
                            parts.append(current_part.strip('"'))
                        
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
    
    def _estimate_total_samples(self) -> int:
        """Estimate total number of samples in the dataset based on split."""
        
        # OPTIMIZATION: Use memory catalog count if available
        if self.memory_catalog:
            if self.split == "train_small":
                count = len([v for v in self.memory_catalog.values() if v.get('split') == 'train_small'])
                print(f"📊 Memory catalog shows {count:,} samples for '{self.split}' split")
                return count
            else:
                count = len(self.memory_catalog)
                print(f"📊 Memory catalog shows {count:,} total samples")
                return count
        
        # Known sample counts for different splits
        split_sizes = {
            'train': 10_988_032,      # Full training set
            'train_small': 953_000,   # Lightweight training set (~953K samples)
        }
        
        # If we have a known split size, use it
        if self.split in split_sizes:
            estimated_size = split_sizes[self.split]
            print(f"📊 Using known size for split '{self.split}': ~{estimated_size:,} samples")
            return estimated_size
        
        # Method 1: Try to get from catalog if available (but adjust for split)
        if hasattr(self, 'catalog_path') and os.path.exists(self.catalog_path):
            try:
                # Count lines in catalog (minus header)
                with open(self.catalog_path, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                
                # If using train_small, the catalog still shows all samples but we need to estimate
                # the actual filtered count. train_small is roughly 8.7% of the full dataset
                if self.split == 'train_small':
                    estimated_filtered = int(line_count * 0.087)  # ~8.7% ratio
                    print(f"📊 Catalog shows {line_count:,} total samples, estimating ~{estimated_filtered:,} for '{self.split}' split")
                    return estimated_filtered
                else:
                    print(f"📊 Catalog indicates ~{line_count:,} total samples for split '{self.split}'")
                    return line_count
            except Exception as e:
                print(f"⚠️  Could not count catalog entries: {e}")
        
        # Method 2: Check if index file exists and count entries (with split adjustment)
        if hasattr(self, 'index_path') and os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    index_count = sum(1 for _ in f)
                
                # Apply split ratio if needed
                if self.split == 'train_small':
                    estimated_filtered = int(index_count * 0.087)
                    print(f"📊 Index shows {index_count:,} entries, estimating ~{estimated_filtered:,} for '{self.split}' split")
                    return estimated_filtered
                else:
                    print(f"📊 Index indicates ~{index_count:,} total samples for split '{self.split}'")
                    return index_count
            except Exception as e:
                print(f"⚠️  Could not count index entries: {e}")
        
        # Method 3: Use known dataset size as fallback
        fallback_size = split_sizes.get(self.split, 10_988_032)
        print(f"📊 Using default size estimate for split '{self.split}': ~{fallback_size:,} samples")
        return fallback_size
    
    def _parse_catalog_line(self, line: str) -> Dict[str, str]:
        """Parse a CSV line manually to extract taxonomy fields."""
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
        
        # Map to taxonomy fields (adjust indices based on catalog structure)
        # Typical order: split,treeoflife_id,eol_content_id,eol_page_id,bioscan_part,bioscan_filename,
        #                inat21_filename,inat21_cls_name,inat21_cls_num,kingdom,phylum,class,order,family,genus,species,common
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

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset without loading all data."""
        info = {
            'dataset_name': 'TreeOfLife-10M',
            'taxonomic_level': self.taxonomic_level,
            'estimated_total_samples': self._estimate_total_samples()
        }
        
        # Add catalog info if available
        if hasattr(self, 'catalog_path') and os.path.exists(self.catalog_path):
            info['catalog_available'] = True
            info['catalog_path'] = self.catalog_path
        else:
            info['catalog_available'] = False
        
        # Add optimization info
        info['optimizations'] = {
            'memory_catalog_loaded': self.memory_catalog is not None,
            'memory_catalog_size': len(self.memory_catalog) if self.memory_catalog else 0,
            'smart_sampling_enabled': self.use_smart_sampling,
            'catalog_cache_size': self.catalog_cache_size
        }
        
        # Add configuration info
        info['config'] = {
            'max_samples': self.max_samples,
            'max_shards': self.max_shards,
            'min_images_per_class': self.min_images_per_class,
            'strict_label_filtering': self.strict_label_filtering,
            'exclude_partial_labels': self.exclude_partial_labels
        }
        
        return info