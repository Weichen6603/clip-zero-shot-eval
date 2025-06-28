"""TreeOfLife-10M dataset adapter with HuggingFace integration for efficient loading."""

import os
import sys
import time
import random
import sqlite3
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from PIL import Image
import torch
import pandas as pd

# Configure PIL to handle large images and suppress warnings
Image.MAX_IMAGE_PIXELS = None  # Remove size limit
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_dataset import BaseDatasetAdapter


class TreeOfLifeAdapter(BaseDatasetAdapter):
    """
    TreeOfLife-10M dataset adapter with HuggingFace integration.
    
    This adapter supports the TreeOfLife-10M dataset which contains over 10 million
    images covering 454,000+ taxa in the tree of life. Uses HuggingFace datasets
    for efficient data loading with configurable shard limits.
    
    Features:
    - HuggingFace streaming for efficient data loading
    - Configurable shard limits (3 shards for lightweight, all for full)
    - Flexible taxonomic level selection (species, genus, family, etc.)
    - Mock data fallback for testing when HuggingFace is unavailable
    """
    
    def __init__(self, root_path: str = "/mnt/d/data/treeoflife", transform=None, split: str = "train", 
                 max_samples: Optional[int] = None, taxonomic_level: str = "species",
                 use_common_names: bool = True, min_images_per_class: int = 1,
                 exclude_partial_labels: bool = False, use_precomputed_embeddings: bool = False,
                 max_shards: Optional[int] = None, **kwargs):
        """
        Initialize TreeOfLife-10M adapter.
        
        Args:
            root_path: Path to dataset cache directory (default: /mnt/d/data/treeoflife)
            transform: Image transformations to apply
            split: Dataset split ('train' only available)
            max_samples: Maximum number of samples to use (None for all)
            taxonomic_level: Level of taxonomy to use for classification 
                           ('species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom')
            use_common_names: Whether to include common names in text templates
            min_images_per_class: Minimum images per class to include class
            exclude_partial_labels: Whether to exclude images without full taxonomic labels
            use_precomputed_embeddings: Whether to use BioCLIP pre-computed text embeddings
            max_shards: Maximum number of data shards to download/use (None for all 73 shards)
        """
        self.max_samples = max_samples
        self.taxonomic_level = taxonomic_level.lower()
        self.use_common_names = use_common_names
        self.min_images_per_class = min_images_per_class
        self.exclude_partial_labels = exclude_partial_labels
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.max_shards = max_shards
        self.catalog_db_path = None  # Will hold SQLite database path
        
        # Validate taxonomic level
        valid_levels = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
        if self.taxonomic_level not in valid_levels:
            raise ValueError(f"taxonomic_level must be one of {valid_levels}")
        
        # Call parent constructor with required parameters
        super().__init__(root_path=root_path, transform=transform, split=split, **kwargs)
    
    def _load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset using HuggingFace with configurable shard limits."""
        print(f"🌳 Loading TreeOfLife-10M data (max_shards: {self.max_shards}, max_samples: {self.max_samples})...")
        
        # Load real data from HuggingFace
        return self._load_real_data()
    
    def _load_real_data(self) -> List[Dict[str, Any]]:
        """Load real TreeOfLife-10M data from HuggingFace with catalog metadata."""
        try:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Run: pip install datasets huggingface_hub")
        
        print("📥 Loading TreeOfLife-10M dataset from HuggingFace...")
        print("⚠️  This may require authentication: huggingface-cli login")
        
        dataset_name = "imageomics/TreeOfLife-10M"
        
        # Step 1: Download the catalog.csv file first
        print("📋 Downloading catalog.csv for taxonomic labels...")
        try:
            catalog_path = hf_hub_download(
                repo_id=dataset_name,
                filename="metadata/catalog.csv",
                cache_dir=self.root_path,
                repo_type="dataset"
            )
            print(f"✅ Downloaded catalog.csv to: {catalog_path}")
            
            # Initialize text-based catalog reader
            self._init_catalog_lazy_reader(catalog_path)
            
        except Exception as e:
            print(f"⚠️  Could not download catalog.csv: {e}")
            print("📡 Falling back to streaming mode without catalog lookup...")
            self.catalog_db_path = None
        
        # Step 2: Load the streaming dataset
        print("🔗 Loading streaming dataset...")
        dataset = load_dataset(
            dataset_name, 
            split="train",
            streaming=True,
            cache_dir=self.root_path
        )
        
        print(f"✅ Successfully connected to TreeOfLife-10M dataset")
        print(f"⏱️  Starting data processing...")
        
        # Convert streaming dataset to samples
        samples = []
        processed_count = 0
        failed_images = []  # Track failed image loading
        catalog_misses = 0  # Track catalog lookup failures
        start_time = time.time()
        
        print("🔍 Processing samples with real taxonomic labels...")
        
        # Estimate total samples for progress tracking
        estimated_total = None
        if self.max_samples:
            estimated_total = min(self.max_samples * 2, 10000)  # Conservative estimate
            print(f"🎯 Target: {self.max_samples} valid samples (scanning up to {estimated_total} total)")
        elif self.max_shards:
            estimated_total = self.max_shards * 140000  # ~140K samples per shard
            print(f"🎯 Processing {self.max_shards} shards (~{estimated_total//1000}K samples)")
        else:
            estimated_total = 10000  # Default estimate for progress display
            print(f"🎯 Processing samples (estimated ~{estimated_total//1000}K samples)")
        
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=estimated_total,
                desc="Processing samples",
                unit="samples",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        except ImportError:
            progress_bar = None
            print("💡 Install tqdm for better progress tracking: pip install tqdm")
        
        for idx, sample in enumerate(dataset):
            processed_count += 1
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                # Update description with current stats
                if processed_count % 100 == 0:
                    progress_bar.set_description(f"Processing samples (found {len(samples)} valid)")
            
            # Check shard limit (approximate)
            if self.max_shards is not None:
                # TreeOfLife-10M has ~140K samples per shard (10M / 73 shards)
                estimated_shard = processed_count // 140000
                if estimated_shard >= self.max_shards:
                    print(f"\n📊 Reached max_shards limit ({self.max_shards}), stopping at sample {processed_count}")
                    break
            
            # Check sample limit
            if self.max_samples is not None and len(samples) >= self.max_samples:
                print(f"\n📊 Reached max_samples limit ({self.max_samples})")
                break
            
            # Extract sample information with robust error handling
            try:
                sample_info = self._process_hf_sample(sample, idx, failed_images)
                if sample_info:
                    samples.append(sample_info)
                else:
                    # Check if it was a catalog miss vs image error
                    sample_id = sample.get('__key__', f'sample_{idx}')
                    if not self._get_taxonomy_from_catalog(sample_id):
                        catalog_misses += 1
                    
                # Early exit if we have enough samples for testing
                if len(samples) >= (self.max_samples or 1000):
                    print(f"\n✅ Found enough samples for evaluation ({len(samples)}), stopping")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ User interrupted at sample {processed_count}")
                break
            except Exception as e:
                if idx < 10:  # Show details for first few errors
                    error_msg = f"⚠️ Critical error processing sample {idx}: {e}"
                    if progress_bar:
                        progress_bar.write(error_msg)
                    else:
                        print(error_msg)
                    # For the first few errors, also print type of error
                    print(f"   Error type: {type(e).__name__}")
                    if hasattr(e, '__traceback__'):
                        import traceback
                        print(f"   Last few stack frames:")
                        traceback.print_exc(limit=3)
                continue
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Display loading statistics
        elapsed_time = time.time() - start_time
        print(f"\n📊 Loading Statistics:")
        print(f"  ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"  📝 Samples processed: {processed_count:,}")
        print(f"  ✅ Valid samples loaded: {len(samples):,}")
        print(f"  🖼️  Image loading failures: {len(failed_images):,}")
        print(f"  📋 Catalog lookup misses: {catalog_misses:,}")
        
        if failed_images:
            print(f"\n⚠️  Image Loading Failures Summary:")
            error_types = {}
            for failure in failed_images[:20]:  # Show first 20 failures
                reason = failure['reason']
                error_type = reason.split(':')[0] if ':' in reason else reason
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {error_type}: {count} failures")
            
            if len(failed_images) > 20:
                print(f"    ... and {len(failed_images) - 20} more failures")
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            print("🔍 Please check:")
            print("  - HuggingFace authentication (huggingface-cli login)")
            print("  - Network connection")
            print("  - Dataset availability")
            raise ValueError("No valid samples extracted from TreeOfLife-10M dataset")
        
        print(f"\n✅ Successfully loaded {len(samples)} valid samples from TreeOfLife-10M")
        return self._filter_samples_by_class_count(samples)
            
    def _process_hf_sample(self, sample: Dict[str, Any], idx: int, failed_images: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Process a single sample from HuggingFace dataset with robust error handling."""
        try:
            # Extract sample ID first for error tracking
            sample_id = sample.get('__key__', f'sample_{idx}')
            sample_url = sample.get('__url__', '')
            
            # Extract image from TreeOfLife-10M format
            image = sample.get('jpg')  # TreeOfLife-10M uses 'jpg' field for images
            if image is None:
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'No jpg field found',
                    'url': sample_url
                })
                return None
            
            # Validate and convert image with comprehensive error handling
            try:
                if not isinstance(image, Image.Image):
                    # Try to convert to PIL Image if it's bytes or other format
                    if hasattr(image, 'read'):
                        import io
                        image_data = image.read()
                        image = Image.open(io.BytesIO(image_data))
                    else:
                        failed_images.append({
                            'sample_id': sample_id,
                            'reason': f'Invalid image type: {type(image)}',
                            'url': sample_url
                        })
                        return None
                
                # Test that we can access basic image properties
                _ = image.size  # This will fail if image is corrupted
                _ = image.mode  # Another basic check
                
                # For corrupted TIFF files, try to convert to RGB mode
                if image.mode not in ['RGB', 'L', 'RGBA']:
                    try:
                        image = image.convert('RGB')
                    except Exception:
                        failed_images.append({
                            'sample_id': sample_id,
                            'reason': f'Cannot convert image mode {image.mode} to RGB',
                            'url': sample_url
                        })
                        return None
                
            except Exception as img_error:
                error_type = type(img_error).__name__
                error_msg = str(img_error)
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': f'{error_type}: {error_msg}',
                    'url': sample_url
                })
                if idx < 5:  # Show details for first few image errors
                    print(f"⚠️  Image loading error for {sample_id}: {error_type} - {error_msg}")
                return None
            
            # Skip extremely large images that might cause memory issues
            try:
                width, height = image.size
                if width * height > 50_000_000:  # 50MP limit
                    failed_images.append({
                        'sample_id': sample_id,
                        'reason': f'Image too large: {width}x{height} pixels',
                        'url': sample_url
                    })
                    if idx < 10:  # Only show warning for first few large images
                        print(f"⚠️  Skipping very large image {sample_id}: {width}x{height} pixels")
                    return None
            except Exception:
                # If we can't even get the size, the image is definitely corrupted
                failed_images.append({
                    'sample_id': sample_id,
                    'reason': 'Cannot access image size - corrupted image',
                    'url': sample_url
                })
                return None
            
            # Try to get real taxonomy from catalog database
            taxonomy = self._get_taxonomy_from_catalog(sample_id)
            
            if taxonomy:
                # Only show first few successful catalog lookups
                if idx < 3:
                    print(f"✅ Found catalog entry for {sample_id}: {taxonomy.get('species', 'unknown')}")
            else:
                # For TreeOfLife, we need real taxonomic data
                if idx < 10:  # Only show warning for first few missing entries
                    print(f"⚠️  No catalog entry found for sample {sample_id}")
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
                'image': image,
                'treeoflife_id': sample_id,
                'metadata': taxonomy
            }
            
            return sample_info
            
        except Exception as e:
            if idx < 10:  # Show details for first few errors only
                print(f"⚠️ Error processing sample {idx}: {e}")
            return None

    def _get_taxonomic_label_from_dict(self, taxonomy: Dict[str, str]) -> Optional[str]:
        """Extract taxonomic label at the specified level from taxonomy dict."""
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
        
        # Return None if label is empty or indicates unknown species
        if not label or label.lower() in ['', 'unknown', 'sp.', 'n/a', 'nan', 'none']:
            return None
        
        # Clean up species labels that have special indicators
        if self.taxonomic_level == 'species' and ('sp.' in label or 'x ' in label):
            # Handle special cases like "sp. ___" or hybrid indicators
            if 'sp.' in label and 'ex' not in label:
                return None  # Skip uncertain species
        
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
        """Get a sample by index with lazy loading and robust image handling."""
        sample_info = self.data[index]
        
        # Get image (already loaded by HuggingFace)
        image = sample_info['image']
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)} for sample {sample_info.get('treeoflife_id', index)}")
        
        # Safety check for image size and integrity
        try:
            width, height = image.size
            # Additional integrity check
            _ = image.mode
            
            if width * height > 50_000_000:  # 50MP limit
                # Resize very large images
                max_size = 2048
                if width > max_size or height > max_size:
                    ratio = min(max_size / width, max_size / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
        except Exception as e:
            print(f"⚠️ Error accessing image properties at index {index} ({sample_info.get('treeoflife_id', 'unknown')}): {e}")
            # Try to create a small placeholder image
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Apply transform if specified
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"⚠️ Error applying transform at index {index} ({sample_info.get('treeoflife_id', 'unknown')}): {e}")
                # Fallback to basic tensor conversion
                import torchvision.transforms as transforms
                to_tensor = transforms.ToTensor()
                try:
                    image = to_tensor(image)
                except Exception as e2:
                    print(f"⚠️ Error in fallback tensor conversion: {e2}")
                    # Create a placeholder tensor
                    import torch
                    image = torch.zeros(3, 224, 224)
        else:
            # Convert PIL image to tensor if no transform is provided
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            try:
                image = to_tensor(image)
            except Exception as e:
                print(f"⚠️ Error converting to tensor at index {index} ({sample_info.get('treeoflife_id', 'unknown')}): {e}")
                # Create a placeholder tensor
                import torch
                image = torch.zeros(3, 224, 224)
        
        # Get label
        label = self.class_to_idx[sample_info['taxonomic_label']]
        
        return image, label
    
    def get_templates(self) -> List[str]:
        """Get text templates for TreeOfLife-10M dataset."""
        from dataset_adapters import DatasetTemplates
        # Use unified templates for consistency across all datasets
        return DatasetTemplates.get_templates()
    
    def _init_catalog_lazy_reader(self, catalog_path: str):
        """Initialize ultra-lightweight catalog reader using text-based indexing."""
        self.catalog_path = catalog_path
        self.catalog_cache = {}  # Minimal cache for recently accessed entries
        self.cache_size = 1000  # Much smaller cache
        
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
    
    def _get_taxonomy_from_catalog(self, sample_id: str) -> Optional[Dict[str, str]]:
        """Ultra-fast taxonomy lookup using text-based index."""
        # Check tiny cache first
        if sample_id in self.catalog_cache:
            return self.catalog_cache[sample_id]
        
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
                        # Parse the line manually
                        taxonomy = self._parse_catalog_line(line)
                        
                        # Add to small cache
                        if len(self.catalog_cache) >= self.cache_size:
                            # Remove half the cache when full
                            keys_to_remove = list(self.catalog_cache.keys())[:self.cache_size//2]
                            for key in keys_to_remove:
                                del self.catalog_cache[key]
                        
                        self.catalog_cache[sample_id] = taxonomy
                        return taxonomy
                        
            return None
            
        except Exception as e:
            print(f"⚠️  Fast lookup error for {sample_id}: {e}")
            return None
    
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
