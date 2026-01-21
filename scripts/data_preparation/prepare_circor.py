#!/usr/bin/env python3
"""
CirCor DigiScope Dataset Preparation Script

Prepares the CirCor DigiScope Phonocardiogram dataset for PA-HCL preprocessing.
Dataset: https://physionet.org/content/circor-heart-sound/

Expected input structure (after downloading):
    circor-heart-sound-1.0.3/
    ├── training_data/
    │   ├── 10001_AV.wav
    │   ├── 10001_MV.wav
    │   ├── 10001_PV.wav
    │   ├── 10001_TV.wav
    │   └── ...
    └── training_data.csv

Output structure (subject-wise):
    data/raw/circor/
    ├── subject_10001/
    │   ├── rec_AV.wav
    │   ├── rec_MV.wav
    │   ├── rec_PV.wav
    │   └── rec_TV.wav
    └── ...
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

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
        description='Prepare CirCor DigiScope dataset for PA-HCL'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to downloaded CirCor dataset directory (e.g., circor-heart-sound-1.0.3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/autodl-tmp/data/raw/circor',
        help='Output directory for organized data (default: /root/autodl-tmp/data/raw/circor)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symbolic links (uses more disk space)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def validate_input_directory(input_dir: Path) -> tuple[Path, Path]:
    """
    Validate input directory structure.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Tuple of (training_data_dir, metadata_file)
        
    Raises:
        FileNotFoundError: If required files/directories are missing
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Check for training_data directory
    training_data_dir = input_dir / "training_data"
    if not training_data_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {training_data_dir}\n"
            "Please ensure you've downloaded and extracted the CirCor dataset correctly."
        )
    
    # Check for metadata CSV
    metadata_file = input_dir / "training_data.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            "The dataset should include training_data.csv"
        )
    
    return training_data_dir, metadata_file


def organize_by_subject(
    training_data_dir: Path,
    metadata_file: Path,
    output_dir: Path,
    copy_files: bool = False
) -> Dict[str, int]:
    """
    Organize CirCor data by subject.
    
    Args:
        training_data_dir: Directory containing .wav files
        metadata_file: Path to training_data.csv
        output_dir: Output directory for organized data
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Dictionary with statistics (num_subjects, num_recordings, etc.)
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_file}")
    df = pd.read_csv(metadata_file)
    logger.info(f"Found {len(df)} patient records in metadata")
    
    # Get all .wav files
    wav_files = list(training_data_dir.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} .wav files")
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {training_data_dir}")
    
    # Group files by subject
    subject_files: Dict[str, List[Path]] = {}
    for wav_file in wav_files:
        # Parse filename: <patient_id>_<location>.wav
        # e.g., 10001_AV.wav -> patient_id=10001, location=AV
        filename = wav_file.stem
        parts = filename.split('_')
        if len(parts) < 2:
            logger.warning(f"Skipping file with unexpected format: {wav_file.name}")
            continue
        
        patient_id = parts[0]
        location = '_'.join(parts[1:])  # Handle multi-part locations
        
        if patient_id not in subject_files:
            subject_files[patient_id] = []
        subject_files[patient_id].append((location, wav_file))
    
    logger.info(f"Organizing {len(subject_files)} subjects")
    
    # Statistics
    stats = {
        'num_subjects': len(subject_files),
        'num_recordings': 0,
        'skipped_files': 0
    }
    
    # Organize files by subject
    for patient_id, recordings in tqdm(subject_files.items(), desc="Organizing subjects"):
        subject_dir = output_dir / f"subject_{patient_id}"
        subject_dir.mkdir(exist_ok=True)
        
        for location, src_file in recordings:
            # Create destination filename: rec_<location>.wav
            dst_file = subject_dir / f"rec_{location}.wav"
            
            try:
                if copy_files:
                    shutil.copy2(src_file, dst_file)
                else:
                    # Create relative symlink
                    if dst_file.exists():
                        dst_file.unlink()
                    dst_file.symlink_to(src_file.resolve())
                
                stats['num_recordings'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {src_file.name}: {e}")
                stats['skipped_files'] += 1
    
    # Save metadata for reference
    metadata_output = output_dir / "circor_metadata.csv"
    df.to_csv(metadata_output, index=False)
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
    logger.info("CirCor DigiScope Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File operation: {'copy' if args.copy else 'symlink'}")
    logger.info("="*60)
    
    try:
        # Validate input
        training_data_dir, metadata_file = validate_input_directory(input_dir)
        
        # Organize data
        stats = organize_by_subject(
            training_data_dir,
            metadata_file,
            output_dir,
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
