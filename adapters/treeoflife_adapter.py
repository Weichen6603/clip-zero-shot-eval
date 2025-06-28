"""TreeOfLife-10M dataset adapter with HuggingFace integration for efficient loading."""

import os
import sys
import time
import random
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from PIL import Image
import torch
import pandas as pd

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
                 max_shards: Optional[int] = None, streaming: bool = True, **kwargs):
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
        # streaming config: explicit arg > kwargs > default True
        if 'streaming' in kwargs:
            self.streaming = kwargs['streaming']
        else:
            self.streaming = streaming
        
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
            
            # Initialize pandas-based lazy loading
            self._init_catalog_lazy_reader(catalog_path)
            
        except Exception as e:
            print(f"⚠️  Could not download catalog.csv: {e}")
            print("📡 Falling back to streaming mode with pseudo-random classification...")
            self.catalog_db_path = None
        
        # Step 2: Load the streaming dataset
        print("🔗 Loading streaming dataset...")
        dataset = load_dataset(
            dataset_name, 
            split="train",
            streaming=self.streaming,
            cache_dir=self.root_path
        )
        
        print(f"✅ Successfully connected to TreeOfLife-10M dataset")
        print(f"⏱️  Starting data processing...")
        
        # Convert streaming dataset to samples
        samples = []
        processed_count = 0
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
            
            # Check sample limit (only if max_samples is not None)
            if self.max_samples is not None and len(samples) >= self.max_samples:
                print(f"\n📊 Reached max_samples limit ({self.max_samples})")
                break
            
            # Extract sample information
            try:
                sample_info = self._process_hf_sample(sample, idx)
                if sample_info:
                    samples.append(sample_info)
                    
                # Early exit for testing (should not trigger if max_samples is None)
                # Remove or comment out the following block:
                # if len(samples) >= (self.max_samples or 1000):
                #     print(f"\n✅ Found enough samples for evaluation ({len(samples)}), stopping")
                #     break
                    
            except Exception as e:
                if idx < 3:  # Show details for first few errors
                    if progress_bar:
                        progress_bar.write(f"⚠️ Error processing sample {idx}: {e}")
                    else:
                        print(f"⚠️ Error processing sample {idx}: {e}")
                continue
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        if not samples:
            print("⚠️ No valid samples found, this might indicate data format issues")
            raise ValueError("No valid samples extracted from TreeOfLife-10M dataset")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Loaded {len(samples)} valid samples from TreeOfLife-10M in {elapsed_time:.1f}s")
        return self._filter_samples_by_class_count(samples)
            
    def _process_hf_sample(self, sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Process a single sample from HuggingFace dataset."""
        try:
            # Extract image from TreeOfLife-10M format
            image = sample.get('jpg')  # TreeOfLife-10M uses 'jpg' field for images
            if image is None:
                return None
            
            # Extract sample ID
            sample_id = sample.get('__key__', f'sample_{idx}')
            sample_url = sample.get('__url__', '')
            
            # Try to get real taxonomy from catalog database
            taxonomy = self._get_taxonomy_from_catalog(sample_id)
            if not taxonomy:
                # Error: cannot proceed without real taxonomic data
                error_msg = (
                    f"❌ Failed to retrieve taxonomic information for sample {sample_id}. "
                    f"This indicates either:\n"
                    f"  1. catalog.csv download failed\n"
                    f"  2. catalog.csv is incomplete or corrupted\n"
                    f"  3. Sample ID format has changed\n"
                    f"TreeOfLife evaluation requires real taxonomic labels and cannot proceed with placeholder data."
                )
                print(error_msg)
                raise ValueError(error_msg)
            
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
            if idx < 3:  # Show details for first few errors only
                print(f"⚠️ Error processing sample {idx}: {e}")
            return None

    def _get_taxonomic_label_from_dict(self, taxonomy: Dict[str, str]) -> Optional[str]:
        """Extract taxonomic label at the specified level from taxonomy dict, skipping confusor/unknown/hybrid labels."""
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
        
        # Return None if label is empty or indicates unknown/confusor/hybrid
        bad_keywords = [
            '', 'unknown', 'sp.', 'n/a', 'nan', 'none', 'confusor', 'confusa', 'confusum', 'confus', 'confuso'
        ]
        label_lower = label.lower()
        if label_lower in bad_keywords:
            return None
        # Also skip if label contains 'sp.' (uncertain species) or ' x ' (hybrid)
        if 'sp.' in label_lower or ' x ' in label_lower:
            return None
        # Clean up species labels that have special indicators
        if self.taxonomic_level == 'species' and ('sp.' in label or 'x ' in label):
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
        """Get a sample by index with lazy loading."""
        sample_info = self.data[index]
        
        # Get image (already loaded by HuggingFace)
        image = sample_info['image']
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Convert PIL image to tensor if no transform is provided
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Get label
        label = self.class_to_idx[sample_info['taxonomic_label']]
        
        return image, label
    
    def get_templates(self) -> List[str]:
        """Get text templates for TreeOfLife-10M dataset."""
        from dataset_adapters import DatasetTemplates
        # Use unified templates for consistency across all datasets
        return DatasetTemplates.get_templates()
    
    def _init_catalog_lazy_reader(self, catalog_path: str):
        """Initialize ultra-lightweight catalog reader using pandas DataFrame for ultra-fast lookup."""
        self.catalog_path = catalog_path
        self.catalog_cache = {}  # Minimal cache for recently accessed entries
        self.cache_size = 10000  # Increase cache size for better performance

        # Create a simple text index if it doesn't exist
        index_path = catalog_path.replace('.csv', '_simple_index.txt')
        if not os.path.exists(index_path) or os.path.getmtime(index_path) < os.path.getmtime(catalog_path):
            print("🔧 Creating ultra-lightweight text index...")
            self._create_simple_index(catalog_path, index_path)
        else:
            print("✅ Using existing text index")
        self.index_path = index_path
        print(f"✅ Ultra-lightweight catalog reader initialized")

        # Load the catalog into a pandas DataFrame for fast lookups
        print("🚀 Loading full catalog.csv into pandas DataFrame (this may take 1-2 minutes)...")
        import pandas as pd
        self._catalog_df = pd.read_csv(catalog_path, index_col='treeoflife_id', low_memory=False)
        print(f"✅ Loaded catalog.csv into DataFrame: {self._catalog_df.shape[0]:,} entries.")
    
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
        """Ultra-fast taxonomy lookup using pandas DataFrame if available, else fallback to index dict."""
        # Check tiny cache first
        if sample_id in self.catalog_cache:
            return self.catalog_cache[sample_id]
        # 优先用 pandas DataFrame 查找
        if hasattr(self, '_catalog_df'):
            try:
                row = self._catalog_df.loc[sample_id]
                taxonomy = {
                    'kingdom': str(row.get('kingdom', '')),
                    'phylum': str(row.get('phylum', '')),
                    'class': str(row.get('class', '')),
                    'order': str(row.get('order', '')),
                    'family': str(row.get('family', '')),
                    'genus': str(row.get('genus', '')),
                    'species': str(row.get('species', '')),
                    'common': str(row.get('common', '')),
                    'split': str(row.get('split', 'train'))
                }
                # Add to cache
                if len(self.catalog_cache) >= self.cache_size:
                    keys_to_remove = list(self.catalog_cache.keys())[:self.cache_size//2]
                    for key in keys_to_remove:
                        del self.catalog_cache[key]
                self.catalog_cache[sample_id] = taxonomy
                return taxonomy
            except KeyError:
                return None
            
        # If DataFrame not available, use ultra-lightweight text index
        if hasattr(self, '_catalog_index_dict') and hasattr(self, '_parse_catalog_line'):
            target_line = self._catalog_index_dict.get(sample_id)
            if target_line is None:
                return None
            try:
                with open(self.catalog_path, 'r', encoding='utf-8') as csv_file:
                    for current_line_num, line in enumerate(csv_file, 1):
                        if current_line_num == target_line:
                            taxonomy = TreeOfLifeAdapter._parse_catalog_line(self, line)
                            if len(self.catalog_cache) >= self.cache_size:
                                keys_to_remove = list(self.catalog_cache.keys())[:self.cache_size//2]
                                for key in keys_to_remove:
                                    del self.catalog_cache[key]
                            self.catalog_cache[sample_id] = taxonomy
                            return taxonomy
                return None
            except Exception as e:
                print(f"⚠️  Fast lookup error for {sample_id}: {e}")
                return None
        return None
    
    def _parse_catalog_row(self, row: pd.Series) -> Dict[str, str]:
        """Parse a pandas Series row to extract taxonomy fields."""
        # Map to taxonomy fields (adjust indices based on catalog structure)
        # Typical order: split,treeoflife_id,eol_content_id,eol_page_id,bioscan_part,bioscan_filename,
        #                inat21_filename,inat21_cls_name,inat21_cls_num,kingdom,phylum,class,order,family,genus,species,common
        return {
            'kingdom': row['kingdom'] if 'kingdom' in row else '',
            'phylum': row['phylum'] if 'phylum' in row else '',
            'class': row['class'] if 'class' in row else '',
            'order': row['order'] if 'order' in row else '',
            'family': row['family'] if 'family' in row else '',
            'genus': row['genus'] if 'genus' in row else '',
            'species': row['species'] if 'species' in row else '',
            'common': row['common'] if 'common' in row else '',
            'split': row['split'] if 'split' in row else 'train'
        }
