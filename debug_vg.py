"""Debug script for Visual Genome dataset issues."""

import sys
import os

# Add parent directory to path
sys.path.append('/home/weichen/Programming/clip-zero-shot-eval')

def debug_visual_genome():
    """Debug Visual Genome dataset loading issues."""
    try:
        from datasets import load_dataset
        
        print("Loading Visual Genome dataset for debugging...")
        dataset = load_dataset(
            "visual_genome", 
            "objects_v1.2.0",
            cache_dir="./data/visual_genome",
            trust_remote_code=True,
            split='train'
        )
        
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset length: {len(dataset) if hasattr(dataset, '__len__') else 'Unknown'}")
        
        # Check first few samples
        print("\nChecking first 5 samples:")
        for i in range(min(5, len(dataset))):
            try:
                sample = dataset[i]
                print(f"\nSample {i}:")
                print(f"  Type: {type(sample)}")
                print(f"  Keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
                
                if isinstance(sample, dict):
                    # Check objects
                    objects = sample.get("objects", [])
                    print(f"  Objects count: {len(objects)}")
                    
                    if objects:
                        obj = objects[0]
                        print(f"  First object type: {type(obj)}")
                        print(f"  First object keys: {obj.keys() if isinstance(obj, dict) else 'Not a dict'}")
                        
                        if isinstance(obj, dict):
                            synsets = obj.get("synsets", [])
                            names = obj.get("names", [])
                            print(f"  First object synsets: {synsets}")
                            print(f"  First object names: {names}")
                
            except Exception as e:
                print(f"Error accessing sample {i}: {e}")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        # Try fallback version
        try:
            print("\nTrying fallback version...")
            dataset = load_dataset(
                "visual_genome", 
                "objects_v1.0.0",
                cache_dir="./data/visual_genome",
                trust_remote_code=True,
                split='train'
            )
            
            print(f"Fallback dataset type: {type(dataset)}")
            print(f"Fallback dataset length: {len(dataset) if hasattr(dataset, '__len__') else 'Unknown'}")
            
            # Check first sample
            sample = dataset[0]
            print(f"\nFallback Sample 0:")
            print(f"  Type: {type(sample)}")
            print(f"  Keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

if __name__ == "__main__":
    debug_visual_genome()
