#!/usr/bin/env python3
"""
Build WebDataset shards for TreeOfLife-10M dataset.

This script converts the TreeOfLife-10M dataset from individual image files
or tar archives into WebDataset format, organizing them by split (train, val, train_small).

WebDataset format stores images and metadata in tar files, enabling:
- Efficient sequential I/O
- Streaming from remote storage
- Built-in shuffling and sharding
- Distributed training support
"""

import os
import sys
import json
import tarfile
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import webdataset as wds
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import hashlib
import logging


class TreeOfLifeWebDatasetBuilder:
    """
    Builds WebDataset shards from TreeOfLife-10M data.
    
    The builder can handle two input formats:
    1. Extracted images: Individual .jpg files named as {treeoflife_id}.jpg
    2. Original tar archives: image_set_*.tar.gz files from HuggingFace
    """
    
    def __init__(
        self,
        input_path: str,
        catalog_path: str,
        output_dir: str,
        shard_size: int = 10000,
        input_format: str = "auto",
        num_workers: int = 1,
        compression: Optional[str] = None
    ):
        """
        Initialize the WebDataset builder.
        
        Args:
            input_path: Path to input data (directory of images or tar files)
            catalog_path: Path to catalog.csv file
            output_dir: Output directory for WebDataset shards
            shard_size: Number of samples per shard file
            input_format: Input format - "images", "tars", or "auto"
            num_workers: Number of parallel workers (currently limited to 1)
            compression: Compression for output shards (None, "gz", "bz2")
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.input_format = input_format
        self.num_workers = num_workers
        self.compression = compression
        
        # Setup logging
        self._setup_logging()
        
        # Load and validate catalog
        self.logger.info(f"Loading catalog from {catalog_path}")
        self.catalog_df = pd.read_csv(catalog_path)
        self.logger.info(f"Loaded {len(self.catalog_df):,} entries from catalog")
        
        # Create ID to metadata mapping
        self._build_metadata_index()
        
        # Detect input format if auto
        if self.input_format == "auto":
            self.input_format = self._detect_input_format()
            self.logger.info(f"Auto-detected input format: {self.input_format}")
        
        # Validate input format
        if self.input_format not in ["images", "tars"]:
            raise ValueError(f"Invalid input format: {self.input_format}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"build_shards_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _detect_input_format(self) -> str:
        """Detect whether input contains images or tar files."""
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.input_path.glob(f"*{ext}")))
        
        # Check for tar files
        tar_files = list(self.input_path.glob("*.tar*"))
        
        if image_files and not tar_files:
            return "images"
        elif tar_files and not image_files:
            return "tars"
        elif image_files and tar_files:
            raise ValueError("Found both images and tar files. Please use only one format.")
        else:
            raise ValueError("No valid input files found.")
    
    def _build_metadata_index(self):
        """Build efficient lookup index for metadata."""
        self.id_to_metadata = {}
        self.split_samples = {
            'train': [],
            'val': [],
            'test': [],
            'train_small': []
        }
        
        for _, row in self.catalog_df.iterrows():
            sample_id = row['treeoflife_id']
            split = row['split']
            
            # Build metadata dictionary
            metadata = {
                'treeoflife_id': sample_id,
                'split': split,
                'scientific_name': str(row.get('scientific_name', '')),
                'common_name': str(row.get('common', '')),
                'kingdom': str(row.get('kingdom', '')),
                'phylum': str(row.get('phylum', '')),
                'class': str(row.get('class', '')),
                'order': str(row.get('order', '')),
                'family': str(row.get('family', '')),
                'genus': str(row.get('genus', '')),
                'species': str(row.get('species', ''))
            }
            
            # Clean metadata - replace 'nan' strings with empty
            for key, value in metadata.items():
                if value.lower() in ['nan', 'none', 'n/a']:
                    metadata[key] = ''
            
            self.id_to_metadata[sample_id] = metadata
            
            # Track samples by split
            if split in self.split_samples:
                self.split_samples[split].append(sample_id)
        
        # Log split statistics
        self.logger.info("Dataset split statistics:")
        for split, samples in self.split_samples.items():
            if samples:
                self.logger.info(f"  {split}: {len(samples):,} samples")
    
    def build(self):
        """Build WebDataset shards for all splits."""
        self.logger.info("Starting WebDataset shard building...")
        self.logger.info(f"Input format: {self.input_format}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Shard size: {self.shard_size:,} samples")
        
        # Process each split
        for split in ['train', 'train_small', 'val', 'test']:
            if self.split_samples[split]:
                self.logger.info(f"\nBuilding shards for {split}...")
                self._build_split_shards(split)
        
        self.logger.info("\nShard building complete!")
        self._write_summary()
    
    def _build_split_shards(self, split: str):
        """Build shards for a specific split."""
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get samples for this split
        split_sample_ids = set(self.split_samples[split])
        
        if self.input_format == "images":
            self._build_from_images(split_sample_ids, split_dir, split)
        else:
            self._build_from_tars(split_sample_ids, split_dir, split)
    
    def _build_from_images(self, sample_ids: set, output_dir: Path, split: str):
        """Build shards from extracted image files."""
        # Collect valid image paths
        valid_samples = []
        missing_count = 0
        
        for sample_id in tqdm(sample_ids, desc=f"Scanning {split} images"):
            # Try different extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = self.input_path / f"{sample_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                valid_samples.append((sample_id, image_path))
            else:
                missing_count += 1
        
        self.logger.info(f"Found {len(valid_samples):,} images, missing {missing_count:,}")
        
        # Write shards
        self._write_shards(valid_samples, output_dir, split, from_images=True)
    
    def _build_from_tars(self, sample_ids: set, output_dir: Path, split: str):
        """Build shards from tar archives."""
        # Find all tar files
        tar_files = sorted(list(self.input_path.glob("*.tar*")))
        self.logger.info(f"Found {len(tar_files)} tar files to process")
        
        # Temporary storage for samples
        samples_data = []
        
        # Process each tar file
        for tar_path in tqdm(tar_files, desc=f"Processing tars for {split}"):
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        
                        # Extract sample ID from filename
                        filename = os.path.basename(member.name)
                        if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                            continue
                        
                        sample_id = os.path.splitext(filename)[0]
                        
                        # Check if this sample belongs to current split
                        if sample_id not in sample_ids:
                            continue
                        
                        # Extract image data
                        f = tar.extractfile(member)
                        if f:
                            image_data = f.read()
                            samples_data.append((sample_id, image_data))
                            f.close()
                            
                            # Write shard if we have enough samples
                            if len(samples_data) >= self.shard_size:
                                self._write_shard_batch(samples_data, output_dir, split)
                                samples_data = []
            
            except Exception as e:
                self.logger.error(f"Error processing {tar_path}: {e}")
        
        # Write remaining samples
        if samples_data:
            self._write_shard_batch(samples_data, output_dir, split)
    
    def _write_shards(self, samples: List[Tuple[str, Path]], output_dir: Path, split: str, from_images: bool = True):
        """Write samples to WebDataset shards."""
        shard_count = 0
        current_shard = []
        
        for sample_id, image_source in tqdm(samples, desc=f"Writing {split} shards"):
            try:
                # Load image data
                if from_images:
                    with open(image_source, 'rb') as f:
                        image_data = f.read()
                else:
                    image_data = image_source  # Already bytes
                
                # Get metadata
                metadata = self.id_to_metadata.get(sample_id, {})
                
                # Add to current shard
                current_shard.append((sample_id, image_data, metadata))
                
                # Write shard if full
                if len(current_shard) >= self.shard_size:
                    self._write_single_shard(current_shard, output_dir, split, shard_count)
                    shard_count += 1
                    current_shard = []
                    
            except Exception as e:
                self.logger.error(f"Error processing {sample_id}: {e}")
        
        # Write remaining samples
        if current_shard:
            self._write_single_shard(current_shard, output_dir, split, shard_count)
            shard_count += 1
        
        self.logger.info(f"Wrote {shard_count} shards for {split}")
    
    def _write_shard_batch(self, samples_data: List[Tuple[str, bytes]], output_dir: Path, split: str):
        """Write a batch of samples as a shard."""
        # Prepare samples with metadata
        shard_samples = []
        for sample_id, image_data in samples_data:
            metadata = self.id_to_metadata.get(sample_id, {})
            shard_samples.append((sample_id, image_data, metadata))
        
        # Determine shard number
        existing_shards = list(output_dir.glob("shard-*.tar*"))
        shard_num = len(existing_shards)
        
        # Write shard
        self._write_single_shard(shard_samples, output_dir, split, shard_num)
    
    def _write_single_shard(self, samples: List[Tuple[str, bytes, dict]], output_dir: Path, split: str, shard_num: int):
        """Write a single shard file."""
        # Determine filename
        if self.compression:
            shard_path = output_dir / f"shard-{shard_num:06d}.tar.{self.compression}"
        else:
            shard_path = output_dir / f"shard-{shard_num:06d}.tar"
        
        # Write shard
        with wds.TarWriter(str(shard_path)) as writer:
            for sample_id, image_data, metadata in samples:
                # Create sample dictionary
                sample = {
                    "__key__": sample_id,
                    "jpg": image_data,
                    "json": json.dumps(metadata).encode('utf-8')
                }
                
                # Add taxonomic path as separate file
                taxon_path = self._build_taxon_path(metadata)
                sample["taxon.txt"] = taxon_path.encode('utf-8')
                
                # Write to shard
                writer.write(sample)
        
        self.logger.debug(f"Wrote shard: {shard_path}")
    
    def _build_taxon_path(self, metadata: dict) -> str:
        """Build taxonomic path string from metadata."""
        levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        parts = []
        
        for level in levels:
            value = metadata.get(level, '').strip()
            if value:
                parts.append(value)
        
        return ' > '.join(parts) if parts else 'Unknown'
    
    def _write_summary(self):
        """Write summary of built shards."""
        summary = {
            'build_date': datetime.now().isoformat(),
            'input_path': str(self.input_path),
            'input_format': self.input_format,
            'output_dir': str(self.output_dir),
            'shard_size': self.shard_size,
            'compression': self.compression,
            'splits': {}
        }
        
        # Count shards per split
        for split in ['train', 'train_small', 'val', 'test']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                shards = list(split_dir.glob("shard-*.tar*"))
                summary['splits'][split] = {
                    'num_shards': len(shards),
                    'expected_samples': len(self.split_samples[split])
                }
        
        # Write summary
        summary_path = self.output_dir / 'build_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Build summary written to {summary_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Build WebDataset shards for TreeOfLife-10M dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from extracted images
  %(prog)s --input-path ./images --catalog ./catalog.csv --output-dir ./webdataset_shards
  
  # Build from tar archives
  %(prog)s --input-path ./tars --catalog ./catalog.csv --output-dir ./webdataset_shards --input-format tars
  
  # Custom shard size with compression
  %(prog)s --input-path ./images --catalog ./catalog.csv --output-dir ./shards --shard-size 5000 --compression gz
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to input data (directory of images or tar files)'
    )
    parser.add_argument(
        '--catalog',
        type=str,
        required=True,
        help='Path to catalog.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for WebDataset shards'
    )
    
    # Optional arguments
    parser.add_argument(
        '--shard-size',
        type=int,
        default=10000,
        help='Number of samples per shard (default: 10000)'
    )
    parser.add_argument(
        '--input-format',
        type=str,
        choices=['auto', 'images', 'tars'],
        default='auto',
        help='Input format (default: auto-detect)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        choices=['gz', 'bz2'],
        help='Compression for output shards (default: none)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers (currently limited to 1)'
    )
    
    args = parser.parse_args()
    
    # Build shards
    try:
        builder = TreeOfLifeWebDatasetBuilder(
            input_path=args.input_path,
            catalog_path=args.catalog,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            input_format=args.input_format,
            num_workers=args.num_workers,
            compression=args.compression
        )
        
        builder.build()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()