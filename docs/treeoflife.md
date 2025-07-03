# TreeOfLife-10M Dataset Guide

## Overview

TreeOfLife-10M is the largest ML-ready dataset of biological organisms, containing over 10 million images covering 454,000+ taxa across the entire tree of life. This guide provides comprehensive instructions for using TreeOfLife-10M with the CLIP zero-shot evaluation framework.

## 🚨 Critical Issue: ID Mismatch

### The Problem
The HuggingFace version of TreeOfLife-10M has a fundamental issue:
```
catalog.csv treeoflife_id: a87c78c0-eb7c-45ee-9a98-681f61292d6b
HuggingFace __key__:       aa563519-6d97-47c3-bcda-c72cdf19a1bf
```
**These IDs are completely different and cannot be matched!**

### Why This Happens
- HuggingFace's WebDataset loader generates random `__key__` values
- Original `treeoflife_id` is not preserved in the uploaded tar files
- No mapping exists between the two ID systems

### Consequences
- ❌ **Streaming mode cannot use real taxonomic labels**
- ❌ **Cannot select exact train_small subset from HuggingFace**
- ❌ **Placeholder labels only in streaming mode**

## Table of Contents

- [Dataset Information](#dataset-information)
- [Three Loading Modes](#three-loading-modes)
- [Correct Solutions](#correct-solutions)
- [Configuration Files](#configuration-files)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [Advanced Configuration](#advanced-configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Dataset Information

### Key Statistics
- **Total Images**: 10,988,032
- **Dataset Splits** (defined in catalog.csv):
  - `train`: ~9.5M images
  - `train_small`: ~953K images (10% subset, recommended for testing)
  - `val`: ~502K images
- **HuggingFace Structure**: Single "train" split containing all images
- **Taxa Coverage**: 454,000+ species across all kingdoms of life
- **Data Sources**: iNaturalist, BIOSCAN-1M, Encyclopedia of Life
- **Storage Requirements**: 
  - Full dataset: ~3TB
  - train_small: ~100GB
  - Catalog metadata: ~500MB

## Three Loading Modes

### Comparison Table

| Feature | Local Mode | WebDataset | Streaming |
|---------|------------|------------|-----------|
| **Requires Download** | Yes (images) | Yes (shards) | No |
| **Accurate Labels** | ✅ Yes | ✅ Yes | ❌ No |
| **Exact train_small** | ✅ Yes | ✅ Yes | ❌ No |
| **Performance** | Medium | Fast | Slow |
| **Storage** | ~100GB | ~100GB | None |
| **Setup Difficulty** | Medium | High | Easy |
| **Research Use** | ✅ Yes | ✅ Yes | ❌ No |

### 1. WebDataset Mode (Recommended for Production)

**Best for**: Large-scale experiments, production deployments, multi-GPU training

**Advantages**:
- Fastest I/O performance
- Supports distributed training
- Efficient memory usage
- Can stream from cloud storage

**Requirements**:
- Pre-built WebDataset shards with proper ID mapping
- ~100GB for train_small shards
- ~3TB for full dataset shards

### 2. Local Mode (Recommended for Development)

**Best for**: Development, debugging, custom data processing

**Advantages**:
- Direct access to individual images
- Perfect ID-to-label matching
- Easy to inspect data
- Flexible for experimentation

**Requirements**:
- Images named as `{treeoflife_id}.jpg` (matching catalog.csv)
- Local storage for images
- catalog.csv file

### 3. Streaming Mode (Limited Use Only)

**Best for**: Quick visualization, checking image quality

**Advantages**:
- No download required
- Minimal storage usage
- Quick to start

**Critical Limitations**:
- ❌ Uses placeholder labels only (class_0, class_1, ...)
- ❌ Cannot match taxonomic information
- ❌ Cannot select exact splits
- ❌ Not suitable for any research or benchmarking

**Note**: Streaming mode does not require downloading `catalog.csv`. It operates independently of the catalog file.

## ✅ Correct Solutions

### Understanding the Limitation

**CRITICAL**: HuggingFace hosts TreeOfLife-10M as a single "train" split containing ALL 11M images. The actual train/train_small/val splits are defined in catalog.csv, but due to ID mismatch, we cannot extract specific splits from HuggingFace.

### Solution 1: Download Original Data (Recommended)

**Option A: Use download script first**
```bash
# Option 1：Download a random subset for the experiment
python download_treeoflife.py --random-subset 10000

# Option 2：Try to download all and match (may fail)
python download_treeoflife.py --download-all
python download_treeoflife.py --download-all --max-images 10000

# Option 3：Only download the catalog and use the streaming mode
python download_treeoflife.py --catalog-only
```

**Option B: Get original data with correct naming**
If the IDs don't match (which is the case), you need the original data:
```bash
# Contact dataset authors for original image_set_*.tar.gz files
# These should contain images already named with treeoflife_id

# Extract preserving original naming
for tar_file in image_set_*.tar.gz; do
    tar -xzf "$tar_file" -C ./data/treeoflife/images/
done

# Verify naming - files should be named like:
# a87c78c0-eb7c-45ee-9a98-681f61292d6b.jpg
ls ./data/treeoflife/images/ | head -5
```

**Step 3: Filter by split (if needed)**
If you downloaded all images, filter to get only your target split:
```python
# filter_split.py - Extract only train_small images
import pandas as pd
import shutil
from pathlib import Path

# Load catalog
catalog = pd.read_csv("./data/treeoflife/metadata/catalog.csv")

# Filter train_small
train_small = catalog[catalog['split'] == 'train_small']
print(f"Found {len(train_small):,} train_small images")

# Copy only train_small images
src_dir = Path("./data/treeoflife/all_images")
dst_dir = Path("./data/treeoflife/images")
dst_dir.mkdir(exist_ok=True)

for _, row in train_small.iterrows():
    src = src_dir / f"{row['treeoflife_id']}.jpg"
    if src.exists():
        shutil.copy(src, dst_dir)
```

**Step 4: Use local mode**
```yaml
# treeoflife_local.yaml
datasets:
  - name: "TreeOfLife-10M-Local"
    type: "treeoflife"
    mode: "local"
    root_path: "./data/treeoflife"
    split: "train_small"
```

### Solution 2: Build Proper WebDataset Shards

**Step 1: Prepare images with correct naming**
```bash
# Must have images named with treeoflife_id
ls images/
# a87c78c0-eb7c-45ee-9a98-681f61292d6b.jpg
# b12d34e5-...jpg
```

**Step 2: Build shards with metadata**
```bash
python build_webdataset_shards.py \
    --input-path ./images \
    --catalog ./metadata/catalog.csv \
    --output-dir ./webdataset_shards \
    --shard-size 10000
```

This creates shards with proper structure:
```
shard-000000.tar
├── a87c78c0-eb7c-45ee-9a98-681f61292d6b.jpg
├── a87c78c0-eb7c-45ee-9a98-681f61292d6b.json  # Contains taxonomy
└── a87c78c0-eb7c-45ee-9a98-681f61292d6b.taxon.txt
```

**Step 3: Use WebDataset mode**
```yaml
# treeoflife_webdataset.yaml
datasets:
  - name: "TreeOfLife-10M-WebDataset"
    type: "treeoflife"
    mode: "webdataset"
    root_path: "./data/treeoflife"
    split: "train_small"
```

### Solution 3: Streaming Mode (Emergency Only)

⚠️ **Only for quick visualization - NO accurate evaluation possible!**

Since we cannot extract specific splits or match IDs, streaming mode can only provide random samples with placeholder labels:

```yaml
# treeoflife_streaming.yaml
datasets:
  - name: "TreeOfLife-10M-Streaming"
    type: "treeoflife"
    mode: "streaming"
    max_samples: 10000  # Random sampling from 11M images
    # Note: ALL settings below are effectively ignored:
    split: "train_small"  # Cannot extract specific split
    taxonomic_level: "species"  # Uses placeholder labels
```

**Alternative: Download random subset for experiments**
```bash
# Download N random images (not a specific split!)
python download_treeoflife.py --random-subset 10000
```

## Configuration Files

We provide three pre-configured YAML files:

### 1. `treeoflife_webdataset.yaml` - WebDataset Configuration
```yaml
# For production use with pre-built shards
datasets:
  - name: "TreeOfLife-10M-WebDataset"
    type: "treeoflife"
    root_path: "./data/treeoflife"
    mode: "webdataset"
    split: "train_small"
    taxonomic_level: "species"
    min_images_per_class: 5
    exclude_partial_labels: false
    strict_label_filtering: false
```

### 2. `treeoflife_local.yaml` - Local Files Configuration
```yaml
# For development with extracted images
datasets:
  - name: "TreeOfLife-10M-Local"
    type: "treeoflife"
    root_path: "./data/treeoflife"
    mode: "local"
    split: "train_small"
    taxonomic_level: "species"
    min_images_per_class: 3
    exclude_partial_labels: false
    strict_label_filtering: false
```

### 3. `treeoflife_streaming.yaml` - Streaming Configuration
```yaml
# For quick tests only - no accurate labels!
datasets:
  - name: "TreeOfLife-10M-Streaming"
    type: "treeoflife"
    root_path: "./data/treeoflife_cache"
    mode: "streaming"
    split: "train_small"  # Note: only affects size, not actual split
    max_samples: 10000
    taxonomic_level: "species"  # Ignored - uses placeholders
```

## Setup Instructions

### Prerequisites

```bash
# Install required packages
pip install torch torchvision clip webdataset pandas huggingface_hub datasets pillow tqdm

# Login to HuggingFace (required for all modes)
huggingface-cli login
```

### Download and Analysis Script

We provide a script to analyze the ID mismatch and attempt downloads:
```bash
# Get the script (save from the provided code)
wget https://raw.githubusercontent.com/your-repo/main/scripts/download_treeoflife.py
# Or create it from the provided script

# Make it executable
chmod +x download_treeoflife.py

# First, analyze if IDs match (they don't)
python download_treeoflife.py --analyze

# Download catalog only
python download_treeoflife.py --catalog-only
```

### Quick Decision Guide

1. **"I just want to see TreeOfLife images"**
   → Download random samples (not a real split)
   ```bash
   python download_treeoflife.py --random-subset 1000
   ```

2. **"I need accurate results for research"**
   → You MUST obtain original data with correct treeoflife_id naming
   ```bash
   # Contact dataset authors or check:
   # - https://treeoflife.cs.cornell.edu/
   # - Original paper authors
   # - Community resources
   ```

3. **"I want to understand the problem"**
   → Run the analysis
   ```bash
   python download_treeoflife.py --analyze
   ```

## Usage Examples

### Basic Usage

```bash
# WebDataset mode (fastest - requires proper shards)
python evaluate_clip.py --config treeoflife_webdataset.yaml

# Local mode (accurate - requires original images)
python evaluate_clip.py --config treeoflife_local.yaml

# Streaming mode (quick test - placeholder labels only!)
python evaluate_clip.py --config treeoflife_streaming.yaml
```

### Custom Parameters

```bash
# Use different taxonomic level
python evaluate_clip.py --config treeoflife_local.yaml \
    --taxonomic_level genus

# Limit samples for testing
python evaluate_clip.py --config treeoflife_local.yaml \
    --max_samples 1000

# Use larger CLIP model
python evaluate_clip.py --config treeoflife_webdataset.yaml \
    --clip_model "ViT-L/14"
```

### Multi-GPU Usage (WebDataset only)

```bash
# Using PyTorch distributed
torchrun --nproc_per_node=4 evaluate_clip.py \
    --config treeoflife_webdataset.yaml
```

## Advanced Configuration

### Taxonomic Levels

You can classify at different levels of the taxonomic hierarchy:

- `species` - Most fine-grained (~450K classes)
- `genus` - Genus level (~100K classes)
- `family` - Family level (~10K classes)
- `order` - Order level (~2K classes)
- `class` - Class level (~500 classes)
- `phylum` - Phylum level (~100 classes)
- `kingdom` - Most coarse-grained (~10 classes)

**Note**: Only works with Local or WebDataset modes!

### Label Quality Control

```yaml
# Exclude samples without complete taxonomic information
exclude_partial_labels: true

# Apply strict filtering to remove uncertain labels
# Filters: "sp.", "cf.", "aff.", "unknown", "complex", etc.
strict_label_filtering: true
```

### Performance Tuning

```yaml
# Batch size (adjust based on GPU memory)
batch_size: 128  # WebDataset can handle larger batches
batch_size: 64   # Local mode typically needs smaller batches
batch_size: 32   # Streaming mode should use small batches

# Number of workers
num_workers: 8   # WebDataset benefits from more workers
num_workers: 4   # Local mode moderate workers
num_workers: 2   # Streaming mode fewer workers

# Class filtering
min_images_per_class: 10  # Higher threshold for cleaner results
min_images_per_class: 1   # Include all classes
```

## Performance Optimization

### Memory Usage

- **WebDataset**: Constant memory usage, scales to any dataset size
- **Local**: Memory usage proportional to catalog size
- **Streaming**: Minimal memory, but no accurate labels

### Speed Optimization

1. **Use WebDataset for production**:
   - Pre-built shards enable fastest loading
   - Supports prefetching and parallel loading

2. **Optimize batch size**:
   ```yaml
   # Find optimal batch size for your GPU
   batch_size: 256  # V100 or better
   batch_size: 128  # RTX 3090
   batch_size: 64   # RTX 3070
   ```

3. **Use multiple workers**:
   ```yaml
   num_workers: 8  # Good for most systems
   ```

### Storage Optimization

- **Full dataset**: Requires ~3TB, use WebDataset with compression
- **train_small**: ~100GB, suitable for most experiments
- **Streaming**: No storage required, but results not valid

## Troubleshooting

### Common Issues

#### "No local images found"
- **Solution**: Download original images with correct naming
- **Quick workaround**: Use streaming mode (but results won't be accurate)

#### "WebDataset shards not found"
- **Solution**: Build shards from properly named images
- **Alternative**: Use local mode with original images

#### ID Mismatch Warning
When using streaming mode, you'll see:
```
=====================================
⚠️  CRITICAL LIMITATION: ID Mismatch
=====================================
HuggingFace __key__ ≠ catalog.csv treeoflife_id
```
This is expected and cannot be fixed. Use local or WebDataset mode for accurate results.

#### Out of Memory
- Reduce `batch_size`
- Use higher taxonomic level (fewer classes)
- Enable `max_samples` to limit dataset size

#### Slow Loading
- Use WebDataset format for best performance
- Increase `num_workers`
- For streaming: check network speed

## Best Practices

1. **Never use streaming mode for research** - only for quick visualization
2. **Always verify image naming** matches treeoflife_id format
3. **Build WebDataset shards** for production use
4. **Document your data source** - critical for reproducibility
5. **Test with small samples first** before full-scale runs

## Research Considerations

For publication-quality results:

1. **MUST use WebDataset or Local mode** (never streaming)
2. **Document exact data source** and preparation steps
3. Enable **strict_label_filtering** for quality control
4. Set **min_images_per_class ≥ 5** for statistical validity
5. Report results at multiple taxonomic levels
6. Include data preparation details in methods section

## Important Notes

1. **HuggingFace hosts only ONE split**: All 11M images are in "train" split
2. **Cannot extract train_small/val**: Due to ID mismatch, specific splits cannot be extracted
3. **Streaming mode uses random sampling**: Not actual train_small subset
4. **Original data required for research**: Must have images named with treeoflife_id
5. **The ID mismatch is unfixable**: This is a fundamental limitation of the HuggingFace upload

## Additional Resources
- [Original Data Sources] - Contact dataset authors
- [Pre-built WebDataset Shards] - Check with community
- [HuggingFace Dataset Card](https://huggingface.co/datasets/imageomics/TreeOfLife-10M)

Remember: The HuggingFace version is convenient for browsing but fundamentally incompatible with the catalog labels. Always verify your data pipeline produces correct treeoflife_id mappings before running experiments!