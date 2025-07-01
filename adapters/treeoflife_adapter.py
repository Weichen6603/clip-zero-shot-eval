"""TreeOfLife-10M dataset adapter with clean architecture and official recommendations."""

import os
import sys
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_dataset import BaseDatasetAdapter


class TreeOfLifeAdapter(BaseDatasetAdapter):
    """
    TreeOfLife-10M dataset adapter following official recommendations.
    
    Supports three modes:
    1. WebDataset (recommended): Use pre-built shards for efficient streaming
    2. Local files: Use extracted images with catalog.csv matching
    3. HuggingFace streaming: Fallback for quick experiments (with limitations)
    
    Dataset splits:
    - train: ~9.5M samples
    - train_small: ~953K samples (10% subset for faster experimentation)
    - val: ~502K samples
    """
    
    def __init__(
        self,
        root_path: str = "./data/treeoflife",
        mode: str = "auto",  # "auto", "webdataset", "local", "streaming"
        split: str = "train_small",
        taxonomic_level: str = "species",
        min_images_per_class: int = 1,
        exclude_partial_labels: bool = False,
        strict_label_filtering: bool = False,
        max_samples: Optional[int] = None,
        transform=None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ):
        """
        Initialize TreeOfLife adapter.
        
        Args:
            root_path: Root directory for dataset
            mode: Data loading mode
            split: Dataset split to use
            taxonomic_level: Taxonomic level for classification
            min_images_per_class: Minimum samples per class
            exclude_partial_labels: Exclude incomplete taxonomic labels
            strict_label_filtering: Apply strict label filtering
            max_samples: Maximum samples to load
            transform: Image transformations
            batch_size: Batch size for DataLoader
            num_workers: Number of data loading workers
        """
        self.root_path = Path(root_path)
        self.mode = mode
        self.split = split
        self.taxonomic_level = taxonomic_level.lower()
        self.min_images_per_class = min_images_per_class
        self.exclude_partial_labels = exclude_partial_labels
        self.strict_label_filtering = strict_label_filtering
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Validate inputs
        self._validate_inputs()
        
        # Detect mode if auto
        if self.mode == "auto":
            self.mode = self._detect_mode()
            print(f"Auto-detected mode: {self.mode}")
        
        # Load catalog
        self.catalog_path = self._ensure_catalog()
        self.catalog_df = None
        
        # Call parent constructor
        super().__init__(
            root_path=str(root_path),
            transform=transform,
            split=split,
            **kwargs
        )
    
    def _validate_inputs(self):
        """Validate input parameters."""
        valid_splits = ['train', 'train_small', 'val', 'test']
        if self.split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {self.split}")
        
        valid_levels = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
        if self.taxonomic_level not in valid_levels:
            raise ValueError(f"taxonomic_level must be one of {valid_levels}, got {self.taxonomic_level}")
        
        valid_modes = ['auto', 'webdataset', 'local', 'streaming']
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")
    
    def _detect_mode(self) -> str:
        """Auto-detect the best available data mode."""
        # Check for WebDataset shards (best option)
        webdataset_dir = self.root_path / "webdataset_shards" / self.split
        if webdataset_dir.exists() and list(webdataset_dir.glob("shard-*.tar")):
            return "webdataset"
        
        # Check for local images
        images_dir = self.root_path / "images"
        if images_dir.exists() and list(images_dir.glob("*.jpg")):
            return "local"
        
        # Default to streaming
        print("No local data found, will use HuggingFace streaming")
        return "streaming"
    
    def _ensure_catalog(self) -> Path:
        """Ensure catalog.csv exists."""
        catalog_path = self.root_path / "metadata" / "catalog.csv"
        
        if not catalog_path.exists():
            print("📥 Downloading catalog.csv...")
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id="imageomics/TreeOfLife-10M",
                    filename="metadata/catalog.csv",
                    cache_dir=str(self.root_path),
                    repo_type="dataset"
                )
                # Copy to expected location
                import shutil
                shutil.copy(downloaded, catalog_path)
                print(f"✅ Catalog downloaded to {catalog_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download catalog: {e}")
        
        return catalog_path
    
    def _load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset based on the configured mode."""
        print(f"\n🌳 Loading TreeOfLife-10M")
        print(f"  Mode: {self.mode}")
        print(f"  Split: {self.split}")
        print(f"  Taxonomic level: {self.taxonomic_level}")
        
        if self.mode == "webdataset":
            return self._load_webdataset()
        elif self.mode == "local":
            return self._load_local()
        elif self.mode == "streaming":
            return self._load_streaming()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _load_webdataset(self) -> List[Dict[str, Any]]:
        """Load using WebDataset format (recommended)."""
        print("📦 Loading WebDataset shards...")
        
        shard_dir = self.root_path / "webdataset_shards" / self.split
        if not shard_dir.exists():
            raise FileNotFoundError(
                f"WebDataset shards not found at {shard_dir}\n"
                f"Please build shards using the official script or switch to 'local' mode"
            )
        
        # Get shard files
        shard_files = sorted(list(shard_dir.glob("shard-*.tar")))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")
        
        print(f"Found {len(shard_files)} shards")
        
        # Create WebDataset
        dataset = wds.WebDataset([str(f) for f in shard_files])
        
        # Process samples
        samples = []
        from tqdm import tqdm
        
        for sample in tqdm(dataset, desc=f"Loading {self.split}"):
            try:
                # Extract image
                image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
                if not image:
                    continue
                
                # Extract metadata
                metadata = {}
                if "json" in sample:
                    metadata = json.loads(sample["json"])
                elif "taxon.txt" in sample:
                    # Parse taxon path
                    taxon_path = sample["taxon.txt"].decode("utf-8").strip()
                    parts = taxon_path.split(" > ")
                    tax_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
                    for i, part in enumerate(parts[:len(tax_levels)]):
                        metadata[tax_levels[i]] = part
                
                # Get label
                label = metadata.get(self.taxonomic_level, "")
                if not self._is_valid_label(label):
                    continue
                
                # Check for partial labels
                if self.exclude_partial_labels and not self._has_full_taxonomy(metadata):
                    continue
                
                # Store sample
                samples.append({
                    "index": len(samples),
                    "taxonomic_label": label,
                    "treeoflife_id": sample.get("__key__", f"sample_{len(samples)}"),
                    "image_bytes": image,
                    "image_path": None,
                    "metadata": metadata
                })
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
                    
            except Exception as e:
                if len(samples) < 10:
                    print(f"Error processing sample: {e}")
                continue
        
        print(f"✅ Loaded {len(samples):,} valid samples")
        return self._filter_by_class_count(samples)
    
    def _load_local(self) -> List[Dict[str, Any]]:
        """Load from local image files."""
        print("💾 Loading from local files...")
        
        # Load catalog for this split
        self.catalog_df = pd.read_csv(self.catalog_path)
        split_df = self.catalog_df[self.catalog_df["split"] == self.split].copy()
        
        print(f"Found {len(split_df):,} entries in catalog for {self.split}")
        
        # Apply sampling if needed
        if self.max_samples and len(split_df) > self.max_samples:
            split_df = split_df.sample(n=self.max_samples, random_state=42)
        
        # Build samples
        samples = []
        images_dir = self.root_path / "images"
        missing_count = 0
        
        from tqdm import tqdm
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Loading images"):
            # Check image exists
            img_path = images_dir / f"{row['treeoflife_id']}.jpg"
            if not img_path.exists():
                missing_count += 1
                continue
            
            # Get label
            label = row.get(self.taxonomic_level)
            if not self._is_valid_label(label):
                continue
            
            # Check for partial labels
            if self.exclude_partial_labels and not self._has_full_taxonomy(row):
                continue
            
            # Build metadata
            metadata = {
                col: str(row.get(col, "")) if pd.notna(row.get(col)) else ""
                for col in ["kingdom", "phylum", "class", "order", "family", "genus", "species", "common"]
            }
            
            samples.append({
                "index": len(samples),
                "taxonomic_label": str(label).strip(),
                "treeoflife_id": row["treeoflife_id"],
                "image_path": str(img_path),
                "image_bytes": None,
                "metadata": metadata
            })
        
        print(f"✅ Loaded {len(samples):,} valid samples")
        print(f"❌ Missing images: {missing_count:,}")
        
        return self._filter_by_class_count(samples)
    
    def _load_streaming(self) -> List[Dict[str, Any]]:
        """Load using HuggingFace streaming (fallback option)."""
        print("📡 Loading via HuggingFace streaming...")
        print("\n" + "="*60)
        print("⚠️  CRITICAL LIMITATION: ID Mismatch")
        print("="*60)
        print("HuggingFace __key__ ≠ catalog.csv treeoflife_id")
        print("Example:")
        print("  HF __key__: aa563519-6d97-47c3-bcda-c72cdf19a1bf")
        print("  Catalog ID: a87c78c0-eb7c-45ee-9a98-681f61292d6b")
        print("\nConsequences:")
        print("  ❌ Cannot match taxonomic labels")
        print("  ❌ Cannot filter exact train_small subset")
        print("  ❌ Will use placeholder labels only")
        print("\nFor accurate evaluation, use 'local' or 'webdataset' mode")
        print("="*60 + "\n")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        # For streaming mode, we'll use sequential sampling for speed
        # Random sampling is too slow with streaming datasets
        target_size = self.max_samples or 10000  # Default to 10K for experiments
        
        print(f"🎯 Loading first {target_size:,} samples (sequential, not from specific split)")
        print("⚡ Using sequential sampling for better performance")
        
        # Load streaming dataset
        dataset = load_dataset(
            "imageomics/TreeOfLife-10M",
            split="train",
            streaming=True,
            cache_dir=str(self.root_path)
        )
        
        samples = []
        errors = 0
        
        from tqdm import tqdm
        pbar = tqdm(total=target_size, desc="Loading samples")
        
        # Take first N valid samples - much faster than random sampling
        for i, sample in enumerate(dataset):
            if len(samples) >= target_size:
                break
            
            try:
                # Get image
                image = sample.get("jpg") or sample.get("image")
                if not image:
                    errors += 1
                    continue
                
                # Convert to bytes efficiently
                if hasattr(image, "save"):
                    import io
                    buffer = io.BytesIO()
                    # Use lower quality for faster processing
                    image.save(buffer, format="JPEG", quality=85)
                    image_bytes = buffer.getvalue()
                else:
                    image_bytes = image
                
                # Create placeholder label
                # Use reasonable number of classes for experiments
                num_placeholder_classes = 100
                class_idx = len(samples) % num_placeholder_classes
                placeholder_label = f"class_{class_idx:03d}"
                
                samples.append({
                    "index": len(samples),
                    "taxonomic_label": placeholder_label,
                    "treeoflife_id": sample.get("__key__", f"sample_{i}"),
                    "image_bytes": image_bytes,
                    "image_path": None,
                    "metadata": {"streaming": True, "hf_index": i}
                })
                
                pbar.update(1)
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n⚠️ Error: {str(e)[:50]}")
                continue
        
        pbar.close()
        
        print(f"\n✅ Loaded {len(samples):,} samples")
        if errors > 0:
            print(f"❌ Skipped {errors:,} samples due to errors")
        print(f"🏷️  Using {num_placeholder_classes} placeholder classes")
        print("\n⚠️  Remember: These are the first N samples, not a specific split!")
        
        return samples
    
    def _is_valid_label(self, label: Any) -> bool:
        """Check if a label is valid."""
        if not label or not isinstance(label, str):
            return False
        
        label = str(label).strip().lower()
        
        # Basic invalid labels
        if label in ['', 'nan', 'none', 'n/a', 'null']:
            return False
        
        if not self.strict_label_filtering:
            return True
        
        # Strict filtering
        uncertain = [
            'unknown', 'sp.', 'spp.', 'cf.', 'aff.', 'nr.',
            'uncertain', 'unidentified', 'complex', 'group'
        ]
        
        return not any(term in label for term in uncertain)
    
    def _has_full_taxonomy(self, data: Union[Dict, pd.Series]) -> bool:
        """Check if all taxonomic levels are present."""
        levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for level in levels:
            value = data.get(level, '')
            if isinstance(data, pd.Series):
                if pd.isna(value) or str(value).strip() == '':
                    return False
            else:
                if not value or str(value).strip() == '':
                    return False
        
        return True
    
    def _filter_by_class_count(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter classes with too few samples."""
        if self.min_images_per_class <= 1:
            return samples
        
        print(f"\n🔍 Filtering classes with < {self.min_images_per_class} samples...")
        
        # Count samples per class
        class_counts = Counter(s['taxonomic_label'] for s in samples)
        
        # Filter
        valid_classes = {c for c, count in class_counts.items() if count >= self.min_images_per_class}
        filtered = [s for s in samples if s['taxonomic_label'] in valid_classes]
        
        print(f"  Classes: {len(class_counts)} → {len(valid_classes)}")
        print(f"  Samples: {len(samples):,} → {len(filtered):,}")
        
        return filtered
    
    def _get_classes(self) -> List[str]:
        """Get list of unique classes."""
        if not self.data:
            return []
        
        classes = sorted(list(set(item['taxonomic_label'] for item in self.data)))
        return classes
    
    def get_dataloader(self) -> DataLoader:
        """Get a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_templates(self) -> List[str]:
        """Get text templates for zero-shot classification."""
        # Biological specimen templates
        return [
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a {} specimen.",
            "a photograph of a {}.",
            "an image of a {}.",
            "a picture of a {}.",
            "a {} in its natural habitat.",
            "a wild {}.",
            "the {} species.",
            "{} in nature.",
            "a specimen of {}.",
            "an example of a {}.",
            "this is a {}.",
            "this is the {}.",
            "this is a {} species.",
        ]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        item = self.data[idx]
        
        # Load image
        if item.get('image_path'):
            image = Image.open(item['image_path']).convert('RGB')
        elif item.get('image_bytes'):
            import io
            image = Image.open(io.BytesIO(item['image_bytes'])).convert('RGB')
        else:
            # Placeholder
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label_idx = self.class_to_idx[item['taxonomic_label']]
        
        return image, label_idx


# Configuration integration
def create_from_config(config: Dict[str, Any]) -> TreeOfLifeAdapter:
    """Create adapter from configuration dictionary."""
    dataset_config = config.get('datasets', [{}])[0]
    
    return TreeOfLifeAdapter(
        root_path=dataset_config.get('root_path', './data/treeoflife'),
        mode='auto',  # Auto-detect best mode
        split=dataset_config.get('split', 'train_small'),
        taxonomic_level=dataset_config.get('taxonomic_level', 'species'),
        min_images_per_class=dataset_config.get('min_images_per_class', 1),
        exclude_partial_labels=dataset_config.get('exclude_partial_labels', False),
        strict_label_filtering=dataset_config.get('strict_label_filtering', False),
        max_samples=dataset_config.get('max_samples'),
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        streaming=dataset_config.get('streaming', False)  # For backward compatibility
    )