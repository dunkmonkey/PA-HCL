#!/usr/bin/env python3
"""
PhysioNet 2016 Challenge Dataset Preparation Script

Prepares the PhysioNet/CinC Challenge 2016 dataset for PA-HCL preprocessing.
Dataset: https://physionet.org/content/challenge-2016/

Expected input structure (after downloading):
    challenge-2016-1.0.0/
    ├── training/
    │   ├── a0001.wav
    │   ├── a0002.wav
    │   └── ...
    ├── validation/
    │   ├── e00001.wav
    │   └── ...
    └── REFERENCE.csv

Output structure (subject-wise):
    data/raw/physionet2016/
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
        description='Prepare PhysioNet 2016 Challenge dataset for PA-HCL'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to downloaded PhysioNet 2016 dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/autodl-tmp/data/raw/physionet2016',
        help='Output directory for organized data (default: /root/autodl-tmp/data/raw/physionet2016)'
    )
    parser.add_argument(
        '--include-validation',
        action='store_true',
        help='Include validation set in addition to training set'
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


def validate_input_directory(input_dir: Path) -> Dict[str, Path]:
    """
    Validate input directory structure.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Dictionary with paths to training/validation directories and reference file
        
    Raises:
        FileNotFoundError: If required files/directories are missing
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    paths = {}
    
    # Check for training directory
    training_dir = input_dir / "training"
    if not training_dir.exists():
        # Try alternative structure (training-a, training-b, etc.)
        training_dirs = list(input_dir.glob("training-*"))
        if training_dirs:
            training_dir = input_dir  # Will process subdirectories later
            paths['training_subdirs'] = training_dirs
        else:
            raise FileNotFoundError(
                f"Training directory not found: {training_dir}\n"
                "Please ensure you've downloaded and extracted the dataset correctly."
            )
    
    paths['training_dir'] = training_dir
    
    # Check for validation directory (optional)
    validation_dir = input_dir / "validation"
    if validation_dir.exists():
        paths['validation_dir'] = validation_dir
    
    # Check for reference file
    reference_file = input_dir / "REFERENCE.csv"
    if not reference_file.exists():
        # Try alternative names
        alt_names = ['reference.csv', 'RECORDS.csv', 'records.csv']
        for alt_name in alt_names:
            alt_file = input_dir / alt_name
            if alt_file.exists():
                reference_file = alt_file
                break
        else:
            logging.warning(
                f"Reference file not found. Expected: {input_dir / 'REFERENCE.csv'}\n"
                "Proceeding without metadata..."
            )
            reference_file = None
    
    paths['reference_file'] = reference_file
    
    return paths


def load_metadata(reference_file: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load metadata from reference file.
    
    Args:
        reference_file: Path to reference CSV file
        
    Returns:
        DataFrame with metadata or None if file doesn't exist
    """
    if reference_file is None or not reference_file.exists():
        return None
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading metadata from {reference_file}")
    
    try:
        # Try different delimiters
        for delimiter in [',', '\t', ' ']:
            try:
                df = pd.read_csv(reference_file, delimiter=delimiter, header=None)
                if len(df.columns) >= 2:
                    # Assume first column is filename, second is label
                    df.columns = ['filename', 'label'] + [f'col_{i}' for i in range(len(df.columns) - 2)]
                    logger.info(f"Loaded {len(df)} records from metadata")
                    return df
            except:
                continue
        
        logger.warning("Could not parse reference file with standard delimiters")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading metadata: {e}")
        return None


def organize_by_subject(
    data_dirs: List[Path],
    output_dir: Path,
    metadata: Optional[pd.DataFrame] = None,
    copy_files: bool = False
) -> Dict[str, int]:
    """
    Organize PhysioNet 2016 data by subject.
    
    Args:
        data_dirs: List of directories containing .wav files
        output_dir: Output directory for organized data
        metadata: Optional metadata DataFrame
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Dictionary with statistics
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all .wav files
    all_wav_files = []
    for data_dir in data_dirs:
        wav_files = list(data_dir.glob("*.wav"))
        all_wav_files.extend(wav_files)
        logger.info(f"Found {len(wav_files)} .wav files in {data_dir}")
    
    if len(all_wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {data_dirs}")
    
    logger.info(f"Total .wav files: {len(all_wav_files)}")
    
    # Statistics
    stats = {
        'num_subjects': 0,
        'num_recordings': 0,
        'skipped_files': 0
    }
    
    # Process each file (each file is typically one subject/recording in this dataset)
    for wav_file in tqdm(all_wav_files, desc="Organizing subjects"):
        # Extract subject ID from filename (e.g., a0001.wav -> a0001)
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
    
    # Save metadata if available
    if metadata is not None:
        metadata_output = output_dir / "physionet2016_metadata.csv"
        metadata.to_csv(metadata_output, index=False)
        logger.info(f"Saved metadata to {metadata_output}")
    
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
    logger.info("PhysioNet 2016 Challenge Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Include validation: {args.include_validation}")
    logger.info(f"File operation: {'copy' if args.copy else 'symlink'}")
    logger.info("="*60)
    
    try:
        # Validate input
        paths = validate_input_directory(input_dir)
        
        # Load metadata
        metadata = load_metadata(paths.get('reference_file'))
        
        # Collect directories to process
        data_dirs = []
        
        # Add training directories
        if 'training_subdirs' in paths:
            data_dirs.extend(paths['training_subdirs'])
        else:
            data_dirs.append(paths['training_dir'])
        
        # Add validation directory if requested
        if args.include_validation and 'validation_dir' in paths:
            data_dirs.append(paths['validation_dir'])
        
        # Organize data
        stats = organize_by_subject(
            data_dirs,
            output_dir,
            metadata=metadata,
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
