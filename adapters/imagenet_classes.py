"""ImageNet class name mappings and utilities."""

# ImageNet-1K class names mapping (synset ID to friendly name)
# This is a subset of the most common classes for demonstration
# In production, you would want to load the complete mapping from LOC_synset_mapping.txt
IMAGENET_CLASS_NAMES = {
    0: "tench",
    1: "goldfish", 
    2: "great white shark",
    3: "tiger shark",
    4: "hammerhead shark",
    5: "electric ray",
    6: "stingray", 
    7: "cock",
    8: "hen",
    9: "ostrich",
    10: "brambling",
    11: "goldfinch",
    12: "house finch", 
    13: "junco",
    14: "indigo bunting",
    15: "robin",
    16: "bulbul",
    17: "jay",
    18: "magpie",
    19: "chickadee"
    # ... (would continue for all 1000 classes)
}

def get_imagenet_class_name(label_idx: int) -> str:
    """Get friendly class name for ImageNet label index.
    
    Args:
        label_idx: Integer label index (0-999)
        
    Returns:
        Friendly class name string
    """
    if label_idx in IMAGENET_CLASS_NAMES:
        return IMAGENET_CLASS_NAMES[label_idx]
    else:
        # Fallback to generic name
        return f"class_{label_idx}"

def load_full_imagenet_mapping():
    """Load complete ImageNet class mapping from external file.
    
    This would load the complete LOC_synset_mapping.txt file in production.
    For now, returns the partial mapping above.
    """
    return IMAGENET_CLASS_NAMES
