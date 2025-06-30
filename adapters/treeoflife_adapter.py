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

    def _filter_samples_by_class_count(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out classes with fewer than min_images_per_class samples."""
        if self.min_images_per_class <= 1:
            return samples

        print(f"\n🔍 Filtering classes with less than {self.min_images_per_class} images...")
        
        class_counts = Counter(sample['taxonomic_label'] for sample in samples)
        
        classes_to_keep = {
            label for label, count in class_counts.items() 
            if count >= self.min_images_per_class
        }
        
        original_count = len(samples)
        original_class_count = len(class_counts)
        
        filtered_samples = [
            sample for sample in samples 
            if sample['taxonomic_label'] in classes_to_keep
        ]
        
        final_class_count = len(classes_to_keep)
        
        print(f"✅ Filtering complete.")
        print(f"  Original classes: {original_class_count}, Final classes: {final_class_count}")
        print(f"  Original samples: {original_count}, Final samples: {len(filtered_samples)}")
        
        return filtered_samples

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        return 10_000_000 # Based on TreeOfLife-10M documentation
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        split_to_load = "train"
        print(f"🔗 Loading streaming dataset (split: {split_to_load})...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split_to_load,
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return [] # Return empty list on failure

        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        start_time = time.time()

        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # Determine target number of samples for the progress bar
        if self.split == "train_small" and self.split_sample_ids:
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} train_small entries")
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            if self.max_samples:
                 print(f"🎯 Target: {self.max_samples} valid samples (scanning up to ~{target_samples:,} total)")
            else:
                 print(f"🎯 Processing full dataset (~{target_samples:,} samples)")

        from tqdm import tqdm
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                if self.max_samples and len(samples) >= self.max_samples:
                    print(f"\n🏁 Reached max_samples limit of {self.max_samples}. Stopping.")
                    break

                sample_id_for_error = 'Unknown'
                try:
                    if not isinstance(sample, dict):
                        print(f"\n⚠️ Skipping non-dict sample of type {type(sample)}")
                        failed_images.append({'sample_id': 'N/A', 'reason': f'Invalid sample format (not a dict): {type(sample)}'})
                        continue

                    sample_id = sample.get('treeoflife_id')
                    if not sample_id:
                        failed_images.append({'sample_id': 'Unknown', 'reason': 'Missing treeoflife_id'})
                        continue
                    
                    sample_id_for_error = sample_id

                    if self.split_sample_ids and sample_id not in self.split_sample_ids:
                        continue

                    metadata = self.memory_catalog.get(sample_id)
                    if metadata is None:
                        catalog_misses += 1
                        continue

                    label = metadata.get(self.taxonomic_level)
                    if not label:
                        continue

                    image_data = sample.get('image')
                    if image_data is None:
                        failed_images.append({'sample_id': sample_id, 'reason': 'Missing image data'})
                        continue

                    image_bytes = self._compress_image(image_data, sample_id)
                    if image_bytes is None:
                        # Error is logged in _compress_image
                        failed_images.append({'sample_id': sample_id, 'reason': 'Image processing/compression failed'})
                        continue
                    
                    processed_sample = {
                        "index": processed_count,
                        "taxonomic_label": label,
                        "treeoflife_id": sample_id,
                        "image_bytes": image_bytes,
                        "metadata": metadata
                    }
                    samples.append(processed_sample)
                    pbar.update(1)
                    pbar.set_postfix(found=len(samples), failed=len(failed_images), refresh=True)

                except Exception as e:
                    print(f"\nUnexpected error processing sample ID {sample_id_for_error}: {e}")
                    failed_images.append({'sample_id': sample_id_for_error, 'reason': str(e)})
                    continue

                processed_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading/processing failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            return []
        
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
            if width * height > 100_000_000: # 100 megapixels limit
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
            image_data = sample.get('jpg')
            if image_data is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None

            # Process and store only compressed image bytes for true streaming
            image_bytes = self._compress_image(image_data, sample_id)
            if image_bytes is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'Image compression/processing failed',
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
        print(f"  Original classes: {original_class_count}, Final classes: {final_class_count}")
        print(f"  Original samples: {original_count}, Final samples: {len(filtered_samples)}")
        
        return filtered_samples

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        # Based on TreeOfLife-10M, the train split has ~10M samples.
        # This is a rough estimate for the progress bar.
        return 10_000_000
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        split_to_load = "train"
        print(f"🔗 Loading streaming dataset (split: {split_to_load})...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split_to_load,
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return [] # Return empty list on failure

        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        start_time = time.time()

        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # Determine target number of samples for the progress bar
        if self.split == "train_small" and self.split_sample_ids:
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} train_small entries")
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            if self.max_samples:
                 print(f"🎯 Target: {self.max_samples} valid samples (scanning up to ~{target_samples:,} total)")
            else:
                 print(f"🎯 Processing full dataset (~{target_samples:,} samples)")

        from tqdm import tqdm
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                if self.max_samples and len(samples) >= self.max_samples:
                    print(f"\n🏁 Reached max_samples limit of {self.max_samples}. Stopping.")
                    break

                sample_id_for_error = 'Unknown'
                try:
                    if not isinstance(sample, dict):
                        print(f"\n⚠️ Skipping non-dict sample of type {type(sample)}")
                        failed_images.append({'sample_id': 'N/A', 'reason': f'Invalid sample format (not a dict): {type(sample)}'})
                        continue

                    sample_id = sample.get('treeoflife_id')
                    if not sample_id:
                        failed_images.append({'sample_id': 'Unknown', 'reason': 'Missing treeoflife_id'})
                        continue
                    
                    sample_id_for_error = sample_id

                    if self.split_sample_ids and sample_id not in self.split_sample_ids:
                        continue

                    metadata = self.memory_catalog.get(sample_id)
                    if metadata is None:
                        catalog_misses += 1
                        continue

                    label = metadata.get(self.taxonomic_level)
                    if not label:
                        continue

                    image_data = sample.get('image')
                    if image_data is None:
                        failed_images.append({'sample_id': sample_id, 'reason': 'Missing image data'})
                        continue

                    image_bytes = self._compress_image(image_data, sample_id)
                    if image_bytes is None:
                        # Error is logged in _compress_image
                        failed_images.append({'sample_id': sample_id, 'reason': 'Image processing/compression failed'})
                        continue
                    
                    processed_sample = {
                        "index": processed_count,
                        "taxonomic_label": label,
                        "treeoflife_id": sample_id,
                        "image_bytes": image_bytes,
                        "metadata": metadata
                    }
                    samples.append(processed_sample)
                    pbar.update(1)
                    pbar.set_postfix(found=len(samples), failed=len(failed_images), refresh=True)

                except Exception as e:
                    print(f"\nUnexpected error processing sample ID {sample_id_for_error}: {e}")
                    failed_images.append({'sample_id': sample_id_for_error, 'reason': str(e)})
                    continue

                processed_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading/processing failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            return []
        
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
            if width * height > 100_000_000: # 100 megapixels limit
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
            image_data = sample.get('jpg')
            if image_data is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None

            # Process and store only compressed image bytes for true streaming
            image_bytes = self._compress_image(image_data, sample_id)
            if image_bytes is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'Image compression/processing failed',
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
        print(f"  Original classes: {original_class_count}, Final classes: {final_class_count}")
        print(f"  Original samples: {original_count}, Final samples: {len(filtered_samples)}")
        
        return filtered_samples

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        # Based on TreeOfLife-10M, the train split has ~10M samples.
        # This is a rough estimate for the progress bar.
        return 10_000_000
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        split_to_load = "train"
        print(f"🔗 Loading streaming dataset (split: {split_to_load})...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split_to_load,
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return [] # Return empty list on failure

        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        start_time = time.time()

        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # Determine target number of samples for the progress bar
        if self.split == "train_small" and self.split_sample_ids:
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} train_small entries")
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            if self.max_samples:
                 print(f"🎯 Target: {self.max_samples} valid samples (scanning up to ~{target_samples:,} total)")
            else:
                 print(f"🎯 Processing full dataset (~{target_samples:,} samples)")

        from tqdm import tqdm
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                if self.max_samples and len(samples) >= self.max_samples:
                    print(f"\n🏁 Reached max_samples limit of {self.max_samples}. Stopping.")
                    break

                sample_id_for_error = 'Unknown'
                try:
                    if not isinstance(sample, dict):
                        print(f"\n⚠️ Skipping non-dict sample of type {type(sample)}")
                        failed_images.append({'sample_id': 'N/A', 'reason': f'Invalid sample format (not a dict): {type(sample)}'})
                        continue

                    sample_id = sample.get('treeoflife_id')
                    if not sample_id:
                        failed_images.append({'sample_id': 'Unknown', 'reason': 'Missing treeoflife_id'})
                        continue
                    
                    sample_id_for_error = sample_id

                    if self.split_sample_ids and sample_id not in self.split_sample_ids:
                        continue

                    metadata = self.memory_catalog.get(sample_id)
                    if metadata is None:
                        catalog_misses += 1
                        continue

                    label = metadata.get(self.taxonomic_level)
                    if not label:
                        continue

                    image_data = sample.get('image')
                    if image_data is None:
                        failed_images.append({'sample_id': sample_id, 'reason': 'Missing image data'})
                        continue

                    image_bytes = self._compress_image(image_data, sample_id)
                    if image_bytes is None:
                        # Error is logged in _compress_image
                        failed_images.append({'sample_id': sample_id, 'reason': 'Image processing/compression failed'})
                        continue
                    
                    processed_sample = {
                        "index": processed_count,
                        "taxonomic_label": label,
                        "treeoflife_id": sample_id,
                        "image_bytes": image_bytes,
                        "metadata": metadata
                    }
                    samples.append(processed_sample)
                    pbar.update(1)
                    pbar.set_postfix(found=len(samples), failed=len(failed_images), refresh=True)

                except Exception as e:
                    print(f"\nUnexpected error processing sample ID {sample_id_for_error}: {e}")
                    failed_images.append({'sample_id': sample_id_for_error, 'reason': str(e)})
                    continue

                processed_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading/processing failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            return []
        
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
            if width * height > 100_000_000: # 100 megapixels limit
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
            image_data = sample.get('jpg')
            if image_data is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None

            # Process and store only compressed image bytes for true streaming
            image_bytes = self._compress_image(image_data, sample_id)
            if image_bytes is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'Image compression/processing failed',
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
        print(f"  Original classes: {original_class_count}, Final classes: {final_class_count}")
        print(f"  Original samples: {original_count}, Final samples: {len(filtered_samples)}")
        
        return filtered_samples

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        # Based on TreeOfLife-10M, the train split has ~10M samples.
        # This is a rough estimate for the progress bar.
        return 10_000_000
    
    def _load_huggingface_data_optimized(self) -> List[Dict[str, Any]]:
        """OPTIMIZED: Load TreeOfLife-10M with smart sampling and memory catalog."""
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace (OPTIMIZED)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        split_to_load = "train"
        print(f"🔗 Loading streaming dataset (split: {split_to_load})...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split_to_load,
                streaming=True,
                cache_dir=self.root_path
            )
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            return [] # Return empty list on failure

        samples = []
        processed_count = 0
        failed_images = []
        catalog_misses = 0
        start_time = time.time()

        print("🔍 Processing samples with OPTIMIZED taxonomic lookups...")
        
        # Determine target number of samples for the progress bar
        if self.split == "train_small" and self.split_sample_ids:
            target_samples = len(self.split_sample_ids)
            if self.max_samples:
                target_samples = min(self.max_samples, target_samples)
            print(f"🎯 Target: {target_samples:,} samples from {len(self.split_sample_ids):,} train_small entries")
        else:
            target_samples = self.max_samples or self._estimate_total_samples()
            if self.max_samples:
                 print(f"🎯 Target: {self.max_samples} valid samples (scanning up to ~{target_samples:,} total)")
            else:
                 print(f"🎯 Processing full dataset (~{target_samples:,} samples)")

        from tqdm import tqdm
        with tqdm(total=target_samples, desc="Processing samples", unit="sample") as pbar:
            for sample in dataset:
                if self.max_samples and len(samples) >= self.max_samples:
                    print(f"\n🏁 Reached max_samples limit of {self.max_samples}. Stopping.")
                    break

                sample_id_for_error = 'Unknown'
                try:
                    if not isinstance(sample, dict):
                        print(f"\n⚠️ Skipping non-dict sample of type {type(sample)}")
                        failed_images.append({'sample_id': 'N/A', 'reason': f'Invalid sample format (not a dict): {type(sample)}'})
                        continue

                    sample_id = sample.get('treeoflife_id')
                    if not sample_id:
                        failed_images.append({'sample_id': 'Unknown', 'reason': 'Missing treeoflife_id'})
                        continue
                    
                    sample_id_for_error = sample_id

                    if self.split_sample_ids and sample_id not in self.split_sample_ids:
                        continue

                    metadata = self.memory_catalog.get(sample_id)
                    if metadata is None:
                        catalog_misses += 1
                        continue

                    label = metadata.get(self.taxonomic_level)
                    if not label:
                        continue

                    image_data = sample.get('image')
                    if image_data is None:
                        failed_images.append({'sample_id': sample_id, 'reason': 'Missing image data'})
                        continue

                    image_bytes = self._compress_image(image_data, sample_id)
                    if image_bytes is None:
                        # Error is logged in _compress_image
                        failed_images.append({'sample_id': sample_id, 'reason': 'Image processing/compression failed'})
                        continue
                    
                    processed_sample = {
                        "index": processed_count,
                        "taxonomic_label": label,
                        "treeoflife_id": sample_id,
                        "image_bytes": image_bytes,
                        "metadata": metadata
                    }
                    samples.append(processed_sample)
                    pbar.update(1)
                    pbar.set_postfix(found=len(samples), failed=len(failed_images), refresh=True)

                except Exception as e:
                    print(f"\nUnexpected error processing sample ID {sample_id_for_error}: {e}")
                    failed_images.append({'sample_id': sample_id_for_error, 'reason': str(e)})
                    continue

                processed_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 OPTIMIZED Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        if elapsed_time > 0:
            print(f"  🚀 Processing speed: {processed_count/elapsed_time:.1f} samples/sec")
        print(f"  🖼️  Image loading/processing failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            return []
        
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
            if width * height > 100_000_000: # 100 megapixels limit
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
            image_data = sample.get('jpg')
            if image_data is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None

            # Process and store only compressed image bytes for true streaming
            image_bytes = self._compress_image(image_data, sample_id)
            if image_bytes is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'Image compression/processing failed',
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
        print(f"  Original classes: {original_class_count}, Final classes: {final_class_count}")
        print(f"  Original samples: {original_count}, Final samples: {len(filtered_samples)}")
        
        return filtered_samples

    def _estimate_total_samples(self) -> int:
        """Provides an estimated total number of samples for progress bars."""
        # Based on TreeOfLife-10M, the train split has ~10M samples.
        # This is a rough estimate for the progress bar.
        return 10_000_000
