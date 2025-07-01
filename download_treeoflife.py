#!/usr/bin/env python3
"""
Download TreeOfLife-10M dataset from HuggingFace.

IMPORTANT LIMITATION:
Due to ID mismatch between HuggingFace (__key__) and catalog.csv (treeoflife_id),
we CANNOT accurately extract specific splits (train_small, val, etc.).

This script provides two approaches:
1. Download ALL images and try to match with catalog (may fail)
2. Download a random subset for experiments (not accurate splits)
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Set, Dict
from tqdm import tqdm
import json
from datetime import datetime
import random


def analyze_id_matching(root_path: str, sample_size: int = 1000):
    """
    Analyze if HuggingFace keys match catalog treeoflife_ids.
    
    Returns:
        bool: True if IDs match, False otherwise
    """
    print("🔍 Analyzing ID matching between HuggingFace and catalog...")
    
    # Load catalog
    catalog_path = Path(root_path) / "metadata" / "catalog.csv"
    if not catalog_path.exists():
        print("❌ Catalog not found. Please download it first.")
        return False
    
    catalog_df = pd.read_csv(catalog_path)
    catalog_ids = set(catalog_df['treeoflife_id'].astype(str))
    print(f"📊 Catalog contains {len(catalog_ids):,} unique treeoflife_ids")
    
    # Sample from HuggingFace
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "imageomics/TreeOfLife-10M",
            split="train",
            streaming=True
        )
        
        hf_keys = []
        print(f"📥 Sampling {sample_size} entries from HuggingFace...")
        
        for i, sample in enumerate(tqdm(dataset, total=sample_size)):
            if i >= sample_size:
                break
            hf_keys.append(sample.get('__key__', ''))
        
        # Check for matches
        matches = sum(1 for key in hf_keys if key in catalog_ids)
        match_rate = matches / len(hf_keys) * 100
        
        print(f"\n📊 Analysis Results:")
        print(f"  HuggingFace samples: {len(hf_keys)}")
        print(f"  Matches found: {matches} ({match_rate:.1f}%)")
        
        if match_rate > 90:
            print("  ✅ IDs appear to match! Can proceed with accurate download.")
            return True
        else:
            print("  ❌ IDs do NOT match. Cannot extract accurate splits.")
            print("\n  Example HF keys:", hf_keys[:3])
            print("  Example catalog IDs:", list(catalog_ids)[:3])
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False


def download_all_with_mapping_attempt(root_path: str, max_images: Optional[int] = None):
    """
    Download all images and attempt to map them to catalog IDs.
    
    This is the most accurate approach but may fail due to ID mismatch.
    """
    root_path = Path(root_path)
    images_dir = root_path / "images_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading all images with mapping attempt...")
    print("⚠️  This will download the ENTIRE dataset unless limited by max_images")
    
    # Load catalog
    catalog_path = root_path / "metadata" / "catalog.csv"
    catalog_df = pd.read_csv(catalog_path)
    catalog_ids = set(catalog_df['treeoflife_id'].astype(str))
    
    try:
        from datasets import load_dataset
        
        # Use streaming to avoid memory issues
        dataset = load_dataset(
            "imageomics/TreeOfLife-10M",
            split="train",  # The ONLY split available
            streaming=True
        )
        
        downloaded = 0
        matched = 0
        id_mapping = {}
        
        pbar = tqdm(total=max_images if max_images else 11_000_000, desc="Downloading")
        
        for sample in dataset:
            if max_images and downloaded >= max_images:
                break
            
            hf_key = sample.get('__key__', f'unknown_{downloaded}')
            
            # Check if this key matches any catalog ID
            if hf_key in catalog_ids:
                # Perfect match!
                treeoflife_id = hf_key
                matched += 1
            else:
                # No match - save with HF key
                treeoflife_id = f"hf_{hf_key}"
            
            # Get image
            image = sample.get('jpg') or sample.get('image')
            if image:
                img_path = images_dir / f"{treeoflife_id}.jpg"
                
                try:
                    if hasattr(image, 'save'):
                        image.save(img_path, 'JPEG', quality=95)
                    else:
                        with open(img_path, 'wb') as f:
                            f.write(image)
                    
                    id_mapping[treeoflife_id] = {
                        'hf_key': hf_key,
                        'matched': hf_key in catalog_ids
                    }
                    
                    downloaded += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️ Error saving image: {e}")
        
        pbar.close()
        
        # Save mapping
        mapping_file = root_path / "download_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(id_mapping, f, indent=2)
        
        print(f"\n📊 Download Summary:")
        print(f"  Total downloaded: {downloaded:,}")
        print(f"  Matched with catalog: {matched:,} ({matched/downloaded*100:.1f}%)")
        print(f"  Images saved to: {images_dir}")
        print(f"  Mapping saved to: {mapping_file}")
        
        if matched < downloaded * 0.9:
            print("\n⚠️  WARNING: Most images could not be matched to catalog IDs!")
            print("  You will need to use alternative approaches.")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def download_random_subset(root_path: str, num_samples: int, split_name: str = "random"):
    """
    Download a random subset of images for experiments.
    
    This does NOT correspond to actual train_small/val splits!
    """
    root_path = Path(root_path)
    images_dir = root_path / f"images_{split_name}_{num_samples}"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {num_samples} random images...")
    print("⚠️  These are NOT from a specific split - just random samples!")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "imageomics/TreeOfLife-10M",
            split="train",
            streaming=True
        )
        
        downloaded = 0
        metadata = []
        
        # Use random sampling
        random.seed(42)  # For reproducibility
        sample_rate = num_samples / 11_000_000  # Approximate total size
        
        pbar = tqdm(total=num_samples, desc="Downloading random subset")
        
        for i, sample in enumerate(dataset):
            # Random sampling
            if random.random() > sample_rate * 2:  # 2x to ensure we get enough
                continue
            
            if downloaded >= num_samples:
                break
            
            hf_key = sample.get('__key__', f'sample_{i}')
            image = sample.get('jpg') or sample.get('image')
            
            if image:
                # Save with sequential naming
                img_path = images_dir / f"{split_name}_{downloaded:08d}.jpg"
                
                try:
                    if hasattr(image, 'save'):
                        image.save(img_path, 'JPEG', quality=95)
                    else:
                        with open(img_path, 'wb') as f:
                            f.write(image)
                    
                    metadata.append({
                        'filename': img_path.name,
                        'hf_key': hf_key,
                        'index': downloaded
                    })
                    
                    downloaded += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️ Error saving image: {e}")
        
        pbar.close()
        
        # Save metadata
        metadata_file = images_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Downloaded {downloaded} random images")
        print(f"  Images saved to: {images_dir}")
        print(f"  Metadata saved to: {metadata_file}")
        print("\n⚠️  Remember: These are random samples, NOT a specific split!")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def download_catalog_only(root_path: str):
    """
    Download only the catalog.csv file. If already exists, notify user and perform data check.
    """
    root_path = Path(root_path)
    catalog_path = root_path / "metadata" / "catalog.csv"

    # If catalog already exists, notify and continue to data check
    if catalog_path.exists():
        print(f"✅ Catalog already exists at {catalog_path}")
        print("🔎 Performing quick data check, please wait...")
    else:
        print("📥 Downloading catalog.csv...")
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="imageomics/TreeOfLife-10M",
                filename="metadata/catalog.csv",
                cache_dir=str(root_path / "cache"),
                repo_type="dataset"
            )
            import shutil
            shutil.copy(downloaded, catalog_path)
            print(f"✅ Catalog saved to {catalog_path}")
            print("🔎 Performing quick data check, please wait...")
        except Exception as e:
            print(f"❌ Failed to download catalog: {e}")
            raise

    # Data check: show split statistics
    try:
        import pandas as pd
        df = pd.read_csv(catalog_path, dtype=str, low_memory=False)
        print("\n📊 Dataset splits in catalog:")
        for split, count in df['split'].value_counts().items():
            print(f"  {split}: {count:,} images")
    except Exception as e:
        print(f"❌ Failed to read catalog for data check: {e}")

    return catalog_path


def main():
    parser = argparse.ArgumentParser(
        description="Download TreeOfLife-10M dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT LIMITATIONS:
  Due to ID mismatch between HuggingFace and catalog.csv,
  we CANNOT accurately extract specific splits (train_small, val, etc.).

Options:
  1. --analyze: Check if IDs match (recommended first step)
  2. --download-all: Download all images and attempt mapping
  3. --random-subset: Download random images for experiments
  4. --catalog-only: Just download catalog.csv

Examples:
  # First, analyze if IDs match
  %(prog)s --analyze
  
  # Download catalog only
  %(prog)s --catalog-only
  
  # Download random 1000 images for testing
  %(prog)s --random-subset 1000
  
  # Attempt to download all and map to catalog (may fail)
  %(prog)s --download-all --max-images 10000
        """
    )
    
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/treeoflife',
        help='Root directory for dataset (default: ./data/treeoflife)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze if HuggingFace IDs match catalog IDs'
    )
    
    parser.add_argument(
        '--catalog-only',
        action='store_true',
        help='Download only catalog.csv'
    )
    
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all images and attempt to map to catalog'
    )
    
    parser.add_argument(
        '--random-subset',
        type=int,
        metavar='N',
        help='Download N random images (not a specific split!)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum images to download with --download-all'
    )
    
    args = parser.parse_args()
    
    # Ensure at least one action is specified
    if not any([args.analyze, args.catalog_only, args.download_all, args.random_subset]):
        print("❌ Please specify at least one action: --analyze, --catalog-only, --download-all, or --random-subset")
        parser.print_help()
        sys.exit(1)
    
    # Check prerequisites
    try:
        import datasets
        import pandas
        import huggingface_hub
        from PIL import Image
    except ImportError as e:
        print("❌ Missing dependencies. Please install:")
        print("   pip install datasets pandas huggingface_hub pillow tqdm")
        sys.exit(1)
    
    # Execute requested actions
    root_path = Path(args.root_path)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Always download catalog first if needed
    if not (root_path / "metadata" / "catalog.csv").exists():
        download_catalog_only(args.root_path)
    
    if args.catalog_only:
        # Ensure catalog is downloaded and perform data check
        download_catalog_only(args.root_path)
        return  # Exit after catalog processing

    if args.analyze:
        can_match = analyze_id_matching(args.root_path)
        if not can_match:
            print("\n💡 Recommendations:")
            print("  1. Use --random-subset for experiments")
            print("  2. Contact dataset authors for original data")
            print("  3. Use streaming mode with placeholder labels")
    
    if args.download_all:
        print("\n" + "="*60)
        print("⚠️  WARNING: Download ALL")
        print("="*60)
        print("This will attempt to download the entire dataset.")
        print("Due to ID mismatch, specific splits cannot be extracted.")
        print("="*60)
        
        response = input("\nContinue? [y/N]: ")
        if response.lower() == 'y':
            download_all_with_mapping_attempt(args.root_path, args.max_images)
        else:
            print("Cancelled.")
    
    if args.random_subset:
        download_random_subset(args.root_path, args.random_subset)


if __name__ == "__main__":
    main()