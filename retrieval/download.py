"""
Download HotpotQA dataset with optional subset for debugging.
"""

import json
import urllib.request
from pathlib import Path


def download_hotpotqa(
    data_dir: Path = Path("data/hotpotqa"),
    debug: bool = False,
    debug_size: int = 100,
):
    """
    Download HotpotQA dataset files.
    
    Args:
        data_dir: Directory to save downloaded files
        debug: If True, only download a subset for debugging
        debug_size: Number of examples to keep in debug mode
        
    Returns:
        List of downloaded file paths
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # HotpotQA dataset URLs
    urls = {
        "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", 
        "hotpot_dev_fullwiki_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    }
    
    downloaded_files = []
    
    for filename, url in urls.items():
        # Always use standard filename - debug mode just subsets the content
        output_filename = filename
            
        filepath = data_dir / output_filename
        
        if filepath.exists():
            print(f"File already exists: {filepath}")
            downloaded_files.append(filepath)
            continue
            
        print(f"Downloading {filename}...")
        try:
            # Download to temporary file first
            temp_filepath = data_dir / f"temp_{filename}"
            urllib.request.urlretrieve(url, temp_filepath)
            
            # Load and optionally subset the data
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
                
            original_size = len(data)
            
            if debug and len(data) > debug_size:
                # Take first debug_size examples
                data = data[:debug_size]
                print(f"  Subsetted from {original_size} to {len(data)} examples for debugging")
            
            # Save the final data
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Remove temp file
            temp_filepath.unlink()
            
            print(f"Downloaded to {filepath}")
            print(f"  Final size: {len(data)} examples")
            downloaded_files.append(filepath)
                
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            # Clean up any partial files
            for cleanup_file in [filepath, data_dir / f"temp_{filename}"]:
                if cleanup_file.exists():
                    cleanup_file.unlink()
            
    print("Download complete!")
    return downloaded_files