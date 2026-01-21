#!/usr/bin/env python3
"""
Unified Data Preparation Script

Main entry point for preparing various heart sound datasets for PA-HCL.
Automatically detects dataset type or allows manual specification.

Usage:
    # Auto-detect dataset type
    python prepare_data.py --input-dir /path/to/dataset --output-dir ./data/raw/organized
    
    # Manually specify dataset type
    python prepare_data.py --dataset-type circor --input-dir /path/to/circor
    
    # Prepare multiple datasets
    python prepare_data.py --dataset-type all --base-dir /path/to/datasets
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Import individual preparation scripts
try:
    from prepare_circor import organize_by_subject as prepare_circor
    from prepare_circor import validate_input_directory as validate_circor
    from prepare_physionet2016 import organize_by_subject as prepare_physionet
    from prepare_physionet2016 import validate_input_directory as validate_physionet
    from prepare_pascal import organize_by_subject as prepare_pascal
    from prepare_pascal import validate_input_directory as validate_pascal
    from prepare_custom import organize_by_subject as prepare_custom
    from prepare_custom import validate_input_directory as validate_custom
except ImportError:
    # If running from different directory, try absolute imports
    import sys
    from pathlib import Path
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    from prepare_circor import organize_by_subject as prepare_circor
    from prepare_circor import validate_input_directory as validate_circor
    from prepare_physionet2016 import organize_by_subject as prepare_physionet
    from prepare_physionet2016 import validate_input_directory as validate_physionet
    from prepare_pascal import organize_by_subject as prepare_pascal
    from prepare_pascal import validate_input_directory as validate_pascal
    from prepare_custom import organize_by_subject as prepare_custom
    from prepare_custom import validate_input_directory as validate_custom


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
        description='Prepare heart sound datasets for PA-HCL preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare CirCor dataset
    python prepare_data.py --dataset-type circor --input-dir /path/to/circor --output-dir data/raw/circor
    
    # Prepare PhysioNet 2016 dataset
    python prepare_data.py --dataset-type physionet2016 --input-dir /path/to/physionet
    
    # Prepare Pascal Challenge dataset
    python prepare_data.py --dataset-type pascal --input-dir /path/to/pascal
    
    # Prepare custom dataset
    python prepare_data.py --dataset-type custom --input-dir data/raw
    
    # Auto-detect dataset type
    python prepare_data.py --input-dir /path/to/dataset
        """
    )
    
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['circor', 'physionet2016', 'pascal', 'custom', 'auto'],
        default='auto',
        help='Type of dataset to prepare (default: auto-detect)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to input dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated based on dataset type)'
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
    
    # Dataset-specific options
    parser.add_argument(
        '--include-validation',
        action='store_true',
        help='[PhysioNet 2016] Include validation set'
    )
    parser.add_argument(
        '--sets',
        nargs='+',
        default=['set_a', 'set_b'],
        help='[Pascal] Dataset subsets to include'
    )
    
    return parser.parse_args()


def detect_dataset_type(input_dir: Path) -> Optional[str]:
    """
    Auto-detect dataset type based on directory structure.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Dataset type string or None if cannot detect
    """
    logger = logging.getLogger(__name__)
    
    # Check for CirCor indicators
    if (input_dir / "training_data").exists() and (input_dir / "training_data.csv").exists():
        logger.info("Detected CirCor DigiScope dataset")
        return 'circor'
    
    # Check for PhysioNet 2016 indicators
    if (input_dir / "training").exists() or list(input_dir.glob("training-*")):
        logger.info("Detected PhysioNet 2016 Challenge dataset")
        return 'physionet2016'
    
    # Check for Pascal Challenge indicators
    if (input_dir / "set_a").exists() or (input_dir / "set_b").exists():
        logger.info("Detected Pascal Challenge dataset")
        return 'pascal'
    
    # Check for custom dataset indicators (Normal/Abnormal folders)
    if (input_dir / "Normal").exists() or (input_dir / "Abnormal").exists():
        logger.info("Detected custom dataset (Normal/Abnormal structure)")
        return 'custom'
    
    # Alternative check for custom dataset
    subdirs = [d.name.lower() for d in input_dir.iterdir() if d.is_dir()]
    if 'normal' in subdirs or 'abnormal' in subdirs:
        logger.info("Detected custom dataset")
        return 'custom'
    
    logger.warning("Could not auto-detect dataset type")
    return None


def prepare_dataset(
    dataset_type: str,
    input_dir: Path,
    output_dir: Path,
    copy_files: bool = False,
    **kwargs
) -> dict:
    """
    Prepare dataset based on type.
    
    Args:
        dataset_type: Type of dataset
        input_dir: Input directory
        output_dir: Output directory
        copy_files: Whether to copy files instead of symlinking
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dictionary with preparation statistics
    """
    logger = logging.getLogger(__name__)
    
    if dataset_type == 'circor':
        logger.info("Preparing CirCor DigiScope dataset...")
        paths = validate_circor(input_dir)
        stats = prepare_circor(
            paths[0],  # training_data_dir
            paths[1],  # metadata_file
            output_dir,
            copy_files=copy_files
        )
        
    elif dataset_type == 'physionet2016':
        logger.info("Preparing PhysioNet 2016 dataset...")
        paths = validate_physionet(input_dir)
        
        # Collect data directories
        data_dirs = []
        if 'training_subdirs' in paths:
            data_dirs.extend(paths['training_subdirs'])
        else:
            data_dirs.append(paths['training_dir'])
        
        if kwargs.get('include_validation') and 'validation_dir' in paths:
            data_dirs.append(paths['validation_dir'])
        
        # Load metadata
        from prepare_physionet2016 import load_metadata
        metadata = load_metadata(paths.get('reference_file'))
        
        stats = prepare_physionet(
            data_dirs,
            output_dir,
            metadata=metadata,
            copy_files=copy_files
        )
        
    elif dataset_type == 'pascal':
        logger.info("Preparing Pascal Challenge dataset...")
        sets = kwargs.get('sets', ['set_a', 'set_b'])
        paths = validate_pascal(input_dir, sets)
        
        # Load labels
        from prepare_pascal import load_labels
        labels = load_labels(paths.get('labels_file'))
        
        stats = prepare_pascal(
            paths['subsets'],
            output_dir,
            labels=labels,
            copy_files=copy_files
        )
        
    elif dataset_type == 'custom':
        logger.info("Preparing custom dataset...")
        paths = validate_custom(input_dir)
        
        # Load metadata
        from prepare_custom import load_metadata
        metadata = load_metadata(paths.get('metadata_file'))
        
        stats = prepare_custom(
            paths['condition_dirs'],
            output_dir,
            metadata=metadata,
            copy_files=copy_files
        )
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return stats


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir).resolve()
    
    # Auto-detect dataset type if needed
    dataset_type = args.dataset_type
    if dataset_type == 'auto':
        dataset_type = detect_dataset_type(input_dir)
        if dataset_type is None:
            logger.error(
                "Could not auto-detect dataset type. "
                "Please specify --dataset-type manually."
            )
            return 1
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(f"/root/autodl-tmp/data/raw/{dataset_type}").resolve()
        logger.info(f"Auto-generated output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir).resolve()
    
    logger.info("="*70)
    logger.info("PA-HCL Data Preparation")
    logger.info("="*70)
    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File operation: {'copy' if args.copy else 'symlink'}")
    logger.info("="*70)
    
    try:
        # Prepare dataset
        stats = prepare_dataset(
            dataset_type,
            input_dir,
            output_dir,
            copy_files=args.copy,
            include_validation=args.include_validation,
            sets=args.sets
        )
        
        # Print summary
        logger.info("="*70)
        logger.info("✓ Data Preparation Complete!")
        logger.info("="*70)
        logger.info(f"Dataset type: {dataset_type}")
        logger.info(f"Total subjects: {stats.get('num_subjects', 'N/A')}")
        logger.info(f"Total recordings: {stats.get('num_recordings', 'N/A')}")
        if stats.get('skipped_files', 0) > 0:
            logger.warning(f"Skipped files: {stats['skipped_files']}")
        if stats.get('parsing_failures', 0) > 0:
            logger.warning(f"Parsing failures: {stats['parsing_failures']}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*70)
        logger.info("\n📝 Next steps:")
        logger.info("1. Review the organized data structure")
        logger.info("2. Run preprocessing:")
        logger.info(f"   python scripts/preprocess.py --config configs/default.yaml")
        logger.info("   (Make sure to update data.raw_dir in config to point to:")
        logger.info(f"    {output_dir})")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during preparation: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
