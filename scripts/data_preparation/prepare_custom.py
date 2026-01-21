#!/usr/bin/env python3
"""
Custom Dataset Preparation Script

Prepares a custom heart sound dataset for PA-HCL preprocessing.
Handles datasets organized by condition (Normal/Abnormal) with files named
using the pattern: <condition>_case<id>_<metadata>_<location>.wav

Expected input structure:
    data/raw/
    ├── Abnormal/
    │   ├── asd_case0001_female_4_20s_USA_A.wav
    │   ├── asd_case0001_female_4_20s_USA_E.wav
    │   ├── asd_case0001_female_4_20s_USA_M.wav
    │   ├── asd_case0001_female_4_20s_USA_P.wav
    │   ├── asd_case0001_female_4_20s_USA_T.wav
    │   └── ...
    ├── Normal/
    │   ├── normal_case0001_male_6_20s_USA_A.wav
    │   ├── normal_case0001_male_6_20s_USA_E.wav
    │   └── ...
    └── metadata.xlsx (optional)

Output structure (subject-wise):
    data/raw/custom_organized/
    ├── subject_asd_case0001/
    │   ├── rec_A.wav
    │   ├── rec_E.wav
    │   ├── rec_M.wav
    │   ├── rec_P.wav
    │   └── rec_T.wav
    ├── subject_normal_case0001/
    │   ├── rec_A.wav
    │   └── ...
    └── metadata.csv
"""

import argparse
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        description='Prepare custom heart sound dataset for PA-HCL'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to input directory containing Normal/ and Abnormal/ folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/autodl-tmp/data/raw/custom_organized',
        help='Output directory for organized data (default: /root/autodl-tmp/data/raw/custom_organized)'
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


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse custom dataset filename.
    
    Expected format: <condition>_case<id>_<gender>_<age>_<duration>_<country>_<location>.wav
    Example: asd_case0001_female_4_20s_USA_A.wav
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary with parsed components or None if parsing fails
    """
    # Remove .wav extension
    stem = filename.replace('.wav', '').replace('.WAV', '')
    
    # Try to match the pattern
    # Pattern: <condition>_case<number>_<...metadata...>_<location>
    # Location is typically a single letter at the end (A, E, M, P, T)
    
    parts = stem.split('_')
    
    if len(parts) < 3:
        return None
    
    # Extract components
    result = {}
    
    # Find case number
    case_idx = None
    for i, part in enumerate(parts):
        if part.startswith('case'):
            case_idx = i
            # Extract case ID
            case_match = re.search(r'case(\d+)', part)
            if case_match:
                result['case_id'] = case_match.group(1)
                result['case_full'] = part
            break
    
    if case_idx is None:
        return None
    
    # Condition is everything before case
    result['condition'] = '_'.join(parts[:case_idx])
    
    # Location is the last part (typically a single letter)
    result['location'] = parts[-1]
    
    # Metadata is everything between case and location
    if case_idx + 1 < len(parts) - 1:
        metadata_parts = parts[case_idx + 1:-1]
        result['metadata'] = '_'.join(metadata_parts)
        
        # Try to extract specific metadata fields
        if len(metadata_parts) >= 1:
            result['gender'] = metadata_parts[0]
        if len(metadata_parts) >= 2:
            result['age'] = metadata_parts[1]
        if len(metadata_parts) >= 3:
            result['duration'] = metadata_parts[2]
        if len(metadata_parts) >= 4:
            result['country'] = metadata_parts[3]
    
    # Create subject ID: <condition>_case<id>
    result['subject_id'] = f"{result['condition']}_case{result['case_id']}"
    
    return result


def validate_input_directory(input_dir: Path) -> Dict[str, Path]:
    """
    Validate input directory structure.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Dictionary with paths to condition folders and metadata
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    paths = {'condition_dirs': []}
    
    # Check for condition directories
    condition_names = ['Normal', 'Abnormal', 'normal', 'abnormal']
    for condition in condition_names:
        condition_dir = input_dir / condition
        if condition_dir.exists() and condition_dir.is_dir():
            paths['condition_dirs'].append(condition_dir)
    
    if not paths['condition_dirs']:
        # Try to find any subdirectories with .wav files
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if list(subdir.glob('*.wav')) or list(subdir.glob('*.WAV')):
                paths['condition_dirs'].append(subdir)
        
        if not paths['condition_dirs']:
            raise FileNotFoundError(
                f"No valid condition directories found in {input_dir}\n"
                "Expected: Normal/ and/or Abnormal/ folders"
            )
    
    # Check for metadata file
    metadata_files = ['metadata.xlsx', 'metadata.csv', 'metadata.xls']
    for metadata_file in metadata_files:
        metadata_path = input_dir / metadata_file
        if metadata_path.exists():
            paths['metadata_file'] = metadata_path
            break
    
    return paths


def load_metadata(metadata_file: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load metadata from Excel or CSV file.
    
    Args:
        metadata_file: Path to metadata file
        
    Returns:
        DataFrame with metadata or None
    """
    if metadata_file is None or not metadata_file.exists():
        return None
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading metadata from {metadata_file}")
    
    try:
        if metadata_file.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(metadata_file)
        else:
            df = pd.read_csv(metadata_file)
        
        logger.info(f"Loaded {len(df)} metadata records")
        return df
    except Exception as e:
        logger.warning(f"Error loading metadata: {e}")
        return None


def organize_by_subject(
    condition_dirs: List[Path],
    output_dir: Path,
    metadata: Optional[pd.DataFrame] = None,
    copy_files: bool = False
) -> Dict[str, int]:
    """
    Organize custom dataset by subject.
    
    Args:
        condition_dirs: List of directories containing .wav files
        output_dir: Output directory for organized data
        metadata: Optional metadata DataFrame
        copy_files: If True, copy files; if False, create symlinks
        
    Returns:
        Dictionary with statistics
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all .wav files and parse filenames
    subject_recordings = defaultdict(list)
    parsing_failures = []
    
    for condition_dir in condition_dirs:
        wav_files = list(condition_dir.glob('*.wav')) + list(condition_dir.glob('*.WAV'))
        logger.info(f"Found {len(wav_files)} .wav files in {condition_dir.name}")
        
        for wav_file in wav_files:
            parsed = parse_filename(wav_file.name)
            
            if parsed is None:
                parsing_failures.append(wav_file.name)
                logger.warning(f"Could not parse filename: {wav_file.name}")
                continue
            
            subject_id = parsed['subject_id']
            location = parsed['location']
            
            subject_recordings[subject_id].append({
                'location': location,
                'src_path': wav_file,
                'parsed': parsed
            })
    
    if parsing_failures:
        logger.warning(f"Failed to parse {len(parsing_failures)} filenames")
        if len(parsing_failures) <= 10:
            for filename in parsing_failures:
                logger.warning(f"  - {filename}")
    
    logger.info(f"Found {len(subject_recordings)} unique subjects")
    
    # Statistics
    stats = {
        'num_subjects': len(subject_recordings),
        'num_recordings': 0,
        'parsing_failures': len(parsing_failures),
        'skipped_files': 0
    }
    
    # Create CSV to store subject metadata
    metadata_records = []
    
    # Organize files by subject
    for subject_id, recordings in tqdm(subject_recordings.items(), desc="Organizing subjects"):
        subject_dir = output_dir / f"subject_{subject_id}"
        subject_dir.mkdir(exist_ok=True)
        
        # Sort recordings by location for consistent ordering
        recordings.sort(key=lambda x: x['location'])
        
        for rec_info in recordings:
            location = rec_info['location']
            src_file = rec_info['src_path']
            parsed = rec_info['parsed']
            
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
                
                # Collect metadata for this recording
                metadata_records.append({
                    'subject_id': subject_id,
                    'case_id': parsed.get('case_id', ''),
                    'condition': parsed.get('condition', ''),
                    'location': location,
                    'gender': parsed.get('gender', ''),
                    'age': parsed.get('age', ''),
                    'duration': parsed.get('duration', ''),
                    'country': parsed.get('country', ''),
                    'original_filename': src_file.name,
                    'organized_path': str(dst_file.relative_to(output_dir))
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {src_file.name}: {e}")
                stats['skipped_files'] += 1
    
    # Save collected metadata
    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_output = output_dir / "custom_metadata.csv"
        metadata_df.to_csv(metadata_output, index=False)
        logger.info(f"Saved metadata to {metadata_output}")
    
    # Also save original metadata if provided
    if metadata is not None:
        original_metadata_output = output_dir / "original_metadata.csv"
        metadata.to_csv(original_metadata_output, index=False)
        logger.info(f"Saved original metadata to {original_metadata_output}")
    
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
    logger.info("Custom Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File operation: {'copy' if args.copy else 'symlink'}")
    logger.info("="*60)
    
    try:
        # Validate input
        paths = validate_input_directory(input_dir)
        
        logger.info(f"Found {len(paths['condition_dirs'])} condition folder(s)")
        for condition_dir in paths['condition_dirs']:
            logger.info(f"  - {condition_dir.name}")
        
        # Load metadata if available
        metadata = load_metadata(paths.get('metadata_file'))
        
        # Organize data
        stats = organize_by_subject(
            paths['condition_dirs'],
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
        logger.info(f"Parsing failures: {stats['parsing_failures']}")
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
