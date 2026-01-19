#!/usr/bin/env python3
"""
Pascal Challenge Dataset Preparation Script

Prepares the Pascal Challenge heart sound dataset for PA-HCL preprocessing.
Dataset: https://istethoscope.peterjbentley.com/heartchallenge/

Expected input structure (after downloading):
    heartchallenge/
    ├── set_a/
    │   ├── a0001.wav
    │   ├── a0002.wav
    │   └── ...
    ├── set_b/
    │   ├── b0001.wav
    │   └── ...
    └── (optional) labels.csv

Output structure (subject-wise):
    data/raw/pascal/
    ├── subject_a0001/
    │   └── rec_01.wav
    ├── subject_a0002/
    │   └── rec_01.wav
    └── ...
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare Pascal Challenge dataset for PA-HCL'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to downloaded Pascal Challenge dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/raw/pascal',
        help='Output directory for organized data (default: ./data/raw/pascal)'
    )
    parser.add_argument(
        '--sets',
        nargs='+',
        default=['set_a', 'set_b'],
        help='Dataset subsets to include (default: set_a set_b)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symbolic links'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def validate_input_directory(input_dir: Path, sets: List[str]) -> Dict[str, Path]:
    """
    Validate input directory structure.
    
    Args:
        input_dir: Path to input directory
        sets: List of subset names to validate
        
    Returns:
        Dictionary with paths to dataset subsets
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    paths = {'subsets': []}
    
    # Check for each subset
    for subset_name in sets:
        subset_dir = input_dir / subset_name
        if subset_dir.exists() and subset_dir.is_dir():
            paths['subsets'].append(subset_dir)
        else:
            # Try alternative naming (e.g., "training_set_a" instead of "set_a")
            alt_names = [
                f"training_{subset_name}",
                f"{subset_name.replace('set_', '')}",
                subset_name.upper(),
                subset_name.lower()
            ]
            found = False
            for alt_name in alt_names:
                alt_dir = input_dir / alt_name
                if alt_dir.exists() and alt_dir.is_dir():
                    paths['subsets'].append(alt_dir)
                    found = True
                    break
            
            if not found:
                logging.warning(f"Subset directory not found: {subset_dir}")
    
    if not paths['subsets']:
        raise FileNotFoundError(
            f"No valid subset directories found in {input_dir}\n"
            f"Expected: {sets}\n"
            "Please check the dataset structure."
        )
    
    # Check for optional labels file
    label_files = ['labels.csv', 'reference.csv', 'REFERENCE.csv']
    for label_file in label_files:
        label_path = input_dir / label_file
        if label_path.exists():
            paths['labels_file'] = label_path
            break
    
    return paths


def load_labels(labels_file: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load labels from file if available.
    
    Args:
        labels_file: Path to labels file
        
    Returns:
        DataFrame with labels or None
    """
    if labels_file is None or not labels_file.exists():
        return None
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading labels from {labels_file}")
    
    try:
        df = pd.read_csv(labels_file)
        logger.info(f"Loaded {len(df)} label records")
        return df
    except Exception as e:
        logger.warning(f"Error loading labels: {e}")
        return None


def organize_by_subject(
    subset_dirs: List[Path],
    output_dir: Path,
    labels: Optional[pd.DataFrame] = None,
    copy_files: bool = False
) -> Dict[str, int]:
    """
    Organize Pascal Challenge data by subject.
    
    Args:
        subset_dirs: List of directories containing .wav files
        output_dir: Output directory for organized data
        labels: Optional labels DataFrame
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Dictionary with statistics
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all .wav files
    all_wav_files = []
    for subset_dir in subset_dirs:
        wav_files = list(subset_dir.glob("*.wav"))
        all_wav_files.extend(wav_files)
        logger.info(f"Found {len(wav_files)} .wav files in {subset_dir.name}")
    
    if len(all_wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {subset_dirs}")
    
    logger.info(f"Total .wav files: {len(all_wav_files)}")
    
    # Statistics
    stats = {
        'num_subjects': 0,
        'num_recordings': 0,
        'skipped_files': 0
    }
    
    # Process each file (each file is typically one subject/recording)
    for wav_file in tqdm(all_wav_files, desc="Organizing subjects"):
        # Extract subject ID from filename
        subject_id = wav_file.stem
        
        # Create subject directory
        subject_dir = output_dir / f"subject_{subject_id}"
        subject_dir.mkdir(exist_ok=True)
        
        # Destination filename
        dst_file = subject_dir / "rec_01.wav"
        
        try:
            if copy_files:
                shutil.copy2(wav_file, dst_file)
            else:
                # Create relative symlink
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(wav_file.resolve())
            
            stats['num_subjects'] += 1
            stats['num_recordings'] += 1
            
        except Exception as e:
            logger.warning(f"Failed to process {wav_file.name}: {e}")
            stats['skipped_files'] += 1
    
    # Save labels if available
    if labels is not None:
        labels_output = output_dir / "pascal_labels.csv"
        labels.to_csv(labels_output, index=False)
        logger.info(f"Saved labels to {labels_output}")
    
    return stats


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    logger.info("="*60)
    logger.info("Pascal Challenge Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset subsets: {args.sets}")
    logger.info(f"File operation: {'copy' if args.copy else 'symlink'}")
    logger.info("="*60)
    
    try:
        # Validate input
        paths = validate_input_directory(input_dir, args.sets)
        
        logger.info(f"Found {len(paths['subsets'])} valid subset(s)")
        for subset in paths['subsets']:
            logger.info(f"  - {subset.name}")
        
        # Load labels if available
        labels = load_labels(paths.get('labels_file'))
        
        # Organize data
        stats = organize_by_subject(
            paths['subsets'],
            output_dir,
            labels=labels,
            copy_files=args.copy
        )
        
        # Print summary
        logger.info("="*60)
        logger.info("Preparation Complete!")
        logger.info("="*60)
        logger.info(f"Total subjects: {stats['num_subjects']}")
        logger.info(f"Total recordings: {stats['num_recordings']}")
        logger.info(f"Skipped files: {stats['skipped_files']}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*60)
        logger.info("\nNext step: Run preprocessing with:")
        logger.info(f"  python scripts/preprocess.py --input-dir {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during preparation: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
