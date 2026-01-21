#!/usr/bin/env python3
"""
Dataset Download Script

Downloads and extracts public heart sound datasets for PA-HCL.
Supports:
- CirCor DigiScope Dataset (from PhysioNet)
- PhysioNet 2016 Challenge Dataset
- Pascal Challenge Dataset (requires manual download)

Usage:
    # Download all datasets
    python download_datasets.py --output-dir ./data/downloads
    
    # Download specific dataset
    python download_datasets.py --dataset circor --output-dir ./data/downloads
    
    # Download without auto-extraction
    python download_datasets.py --dataset physionet2016 --no-extract

Requirements:
    pip install requests tqdm
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install requests tqdm")
    sys.exit(1)


# Dataset information
DATASETS = {
    'circor': {
        'name': 'CirCor DigiScope Dataset',
        'url': 'https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip',
        'filename': 'circor-heart-sound-1.0.3.zip',
        'size': '~10 GB',
        'md5': None,  # Optional: add MD5 checksum if available
        'extract_dir': 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    },
    'physionet2016': {
        'name': 'PhysioNet 2016 Challenge Dataset',
        'url': 'https://physionet.org/static/published-projects/challenge-2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0.zip',
        'filename': 'challenge-2016-1.0.0.zip',
        'size': '~1.2 GB',
        'md5': None,
        'extract_dir': 'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0'
    },
    'pascal': {
        'name': 'Pascal Challenge Dataset',
        'url': 'manual',  # Requires manual download
        'filename': 'heartchallenge.zip',  # Expected filename
        'size': '~500 MB',
        'md5': None,
        'extract_dir': 'heartchallenge',
        'instructions': """
Pascal Challenge Dataset requires manual download:
1. Visit: https://istethoscope.peterjbentley.com/heartchallenge/index.html
2. Register and download the dataset
3. Save as: {output_dir}/heartchallenge.zip
4. Run this script again with --dataset pascal --extract-only
"""
    }
}


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download public heart sound datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['all', 'circor', 'physionet2016', 'pascal'],
        default='all',
        help='Dataset to download (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/downloads',
        help='Output directory for downloads (default: ./data/downloads)'
    )
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Download only, do not extract archives'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Extract existing archives only, skip download'
    )
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='Keep archive file after extraction'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save downloaded file
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Downloading from: {url}")
        logger.info(f"Saving to: {output_path}")
        
        # Send HEAD request to get file size
        response = requests.head(url, allow_redirects=True, timeout=10)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Download complete: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """
    Verify file MD5 checksum.
    
    Args:
        file_path: Path to file
        expected_md5: Expected MD5 hash
        
    Returns:
        True if checksum matches, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Verifying checksum for {file_path.name}...")
    
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    
    if actual_md5 == expected_md5:
        logger.info("Checksum verification passed")
        return True
    else:
        logger.error(f"Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """
    Extract zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting {archive_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Extract with progress bar
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc="Extracting") as pbar:
                    for member in members:
                        zip_ref.extract(member, extract_dir)
                        pbar.update(1)
        
        elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                members = tar_ref.getmembers()
                with tqdm(total=len(members), desc="Extracting") as pbar:
                    for member in members:
                        tar_ref.extract(member, extract_dir)
                        pbar.update(1)
        
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"Extraction complete: {extract_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def process_dataset(
    dataset_key: str,
    output_dir: Path,
    download: bool = True,
    extract: bool = True,
    keep_archive: bool = False
) -> bool:
    """
    Download and/or extract a dataset.
    
    Args:
        dataset_key: Dataset identifier
        output_dir: Output directory
        download: Whether to download
        extract: Whether to extract
        keep_archive: Whether to keep archive after extraction
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if dataset_key not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        return False
    
    dataset_info = DATASETS[dataset_key]
    
    logger.info("="*60)
    logger.info(f"Processing: {dataset_info['name']}")
    logger.info(f"Size: {dataset_info['size']}")
    logger.info("="*60)
    
    archive_path = output_dir / dataset_info['filename']
    
    # Handle manual download datasets
    if dataset_info['url'] == 'manual':
        if download:
            instructions = dataset_info.get('instructions', '').format(output_dir=output_dir)
            logger.info(instructions)
            return False
        
        if not archive_path.exists():
            logger.error(f"Archive not found: {archive_path}")
            logger.info(dataset_info.get('instructions', '').format(output_dir=output_dir))
            return False
    
    # Download
    if download and dataset_info['url'] != 'manual':
        if archive_path.exists():
            logger.info(f"Archive already exists: {archive_path}")
            user_input = input("Re-download? (y/N): ").strip().lower()
            if user_input != 'y':
                logger.info("Skipping download")
            else:
                archive_path.unlink()
                if not download_file(dataset_info['url'], archive_path):
                    return False
        else:
            if not download_file(dataset_info['url'], archive_path):
                return False
        
        # Verify checksum if available
        if dataset_info['md5']:
            if not verify_checksum(archive_path, dataset_info['md5']):
                logger.error("Checksum verification failed. File may be corrupted.")
                return False
    
    # Extract
    if extract:
        extract_base_dir = output_dir / 'extracted'
        
        if not extract_archive(archive_path, extract_base_dir):
            return False
        
        # Remove archive if requested
        if not keep_archive and archive_path.exists():
            logger.info(f"Removing archive: {archive_path}")
            archive_path.unlink()
        
        # Show extracted directory
        extracted_dir = extract_base_dir / dataset_info['extract_dir']
        if extracted_dir.exists():
            logger.info(f"Dataset extracted to: {extracted_dir}")
            logger.info("\nNext step: Prepare dataset with:")
            logger.info(f"  python scripts/data_preparation/prepare_{dataset_key}.py \\")
            logger.info(f"    --input-dir {extracted_dir} \\")
            logger.info(f"    --output-dir /root/autodl-tmp/data/raw/{dataset_key}")
        else:
            logger.warning(f"Expected directory not found: {extracted_dir}")
            logger.info(f"Check contents of: {extract_base_dir}")
    
    return True


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine what to download
    if args.dataset == 'all':
        datasets_to_process = ['circor', 'physionet2016', 'pascal']
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    results = {}
    for dataset_key in datasets_to_process:
        success = process_dataset(
            dataset_key,
            output_dir,
            download=not args.extract_only,
            extract=not args.no_extract,
            keep_archive=args.keep_archive
        )
        results[dataset_key] = success
        print()  # Blank line between datasets
    
    # Print summary
    logger.info("="*60)
    logger.info("Download Summary")
    logger.info("="*60)
    for dataset_key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{DATASETS[dataset_key]['name']}: {status}")
    logger.info("="*60)
    
    # Return error code if any failed
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
