#!/usr/bin/env python3
"""
Test script for data preparation scripts

This script performs unit tests and integration tests for all data preparation scripts.

Usage:
    python test_data_preparation.py
    python test_data_preparation.py --verbose
    python test_data_preparation.py --test-download  # Includes download tests (slow)
"""

import argparse
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pandas as pd
    import numpy as np
    from scipy.io import wavfile
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install pandas numpy scipy")
    sys.exit(1)

# Import preparation modules
try:
    from prepare_circor import parse_args as circor_args, organize_by_subject as circor_organize
    from prepare_physionet2016 import parse_args as physio_args, organize_by_subject as physio_organize
    from prepare_pascal import parse_args as pascal_args, organize_by_subject as pascal_organize
    from prepare_custom import parse_args as custom_args, parse_filename, organize_by_subject as custom_organize
except ImportError as e:
    print(f"Error importing preparation modules: {e}")
    sys.exit(1)


class TestFilenameParser(unittest.TestCase):
    """Test custom dataset filename parsing."""
    
    def test_standard_format(self):
        """Test parsing of standard format filenames."""
        filename = "asd_case0001_female_4_20s_USA_A.wav"
        result = parse_filename(filename)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['case_id'], '0001')
        self.assertEqual(result['condition'], 'asd')
        self.assertEqual(result['location'], 'A')
        self.assertEqual(result['gender'], 'female')
        self.assertEqual(result['age'], '4')
        self.assertEqual(result['subject_id'], 'asd_case0001')
    
    def test_normal_format(self):
        """Test parsing of normal subject filenames."""
        filename = "normal_case0123_male_35_15s_CHN_M.wav"
        result = parse_filename(filename)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['case_id'], '0123')
        self.assertEqual(result['condition'], 'normal')
        self.assertEqual(result['location'], 'M')
        self.assertEqual(result['subject_id'], 'normal_case0123')
    
    def test_invalid_format(self):
        """Test parsing of invalid filenames."""
        invalid_filenames = [
            "invalid.wav",
            "no_case_here.wav",
            "case0001.wav"
        ]
        
        for filename in invalid_filenames:
            result = parse_filename(filename)
            self.assertIsNone(result, f"Should fail to parse: {filename}")


class TestDatasetOrganization(unittest.TestCase):
    """Test dataset organization functions."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_dummy_wav(self, filepath: Path, duration: float = 1.0, sample_rate: int = 4000):
        """Create a dummy WAV file for testing."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int(duration * sample_rate)
        # Generate simple sine wave
        t = np.linspace(0, duration, num_samples)
        data = np.sin(2 * np.pi * 60 * t) * 0.5  # 60 Hz tone
        data_int = (data * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, data_int)
    
    def test_circor_organization(self):
        """Test CirCor dataset organization."""
        # Create dummy input structure
        training_dir = self.input_dir / "training_data"
        training_dir.mkdir()
        
        # Create dummy files
        self.create_dummy_wav(training_dir / "10001_AV.wav")
        self.create_dummy_wav(training_dir / "10001_MV.wav")
        self.create_dummy_wav(training_dir / "10002_AV.wav")
        
        # Create dummy metadata
        metadata_file = self.input_dir / "training_data.csv"
        df = pd.DataFrame({
            'Patient ID': ['10001', '10002'],
            'Age': [25, 30],
            'Sex': ['Male', 'Female']
        })
        df.to_csv(metadata_file, index=False)
        
        # Run organization
        stats = circor_organize(
            training_dir,
            metadata_file,
            self.output_dir,
            copy_files=True
        )
        
        # Verify output
        self.assertEqual(stats['num_subjects'], 2)
        self.assertEqual(stats['num_recordings'], 3)
        
        # Check directory structure
        subject_10001 = self.output_dir / "subject_10001"
        self.assertTrue(subject_10001.exists())
        self.assertTrue((subject_10001 / "rec_AV.wav").exists())
        self.assertTrue((subject_10001 / "rec_MV.wav").exists())
    
    def test_custom_organization(self):
        """Test custom dataset organization."""
        # Create dummy input structure
        normal_dir = self.input_dir / "Normal"
        abnormal_dir = self.input_dir / "Abnormal"
        normal_dir.mkdir()
        abnormal_dir.mkdir()
        
        # Create dummy files with proper naming
        self.create_dummy_wav(normal_dir / "normal_case0001_male_25_10s_USA_A.wav")
        self.create_dummy_wav(normal_dir / "normal_case0001_male_25_10s_USA_M.wav")
        self.create_dummy_wav(abnormal_dir / "asd_case0001_female_4_15s_USA_A.wav")
        
        # Run organization
        stats = custom_organize(
            [normal_dir, abnormal_dir],
            self.output_dir,
            copy_files=True
        )
        
        # Verify output
        self.assertEqual(stats['num_subjects'], 2)
        self.assertEqual(stats['num_recordings'], 3)
        
        # Check directory structure
        subject_normal = self.output_dir / "subject_normal_case0001"
        subject_asd = self.output_dir / "subject_asd_case0001"
        
        self.assertTrue(subject_normal.exists())
        self.assertTrue(subject_asd.exists())
        self.assertTrue((subject_normal / "rec_A.wav").exists())
        self.assertTrue((subject_normal / "rec_M.wav").exists())


class TestWAVFileValidation(unittest.TestCase):
    """Test WAV file validation and creation."""
    
    def setUp(self):
        """Set up temporary directory."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_valid_wav(self):
        """Test creation of valid WAV files."""
        filepath = Path(self.test_dir) / "test.wav"
        sample_rate = 4000
        duration = 2.0
        
        # Create WAV
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        data = np.sin(2 * np.pi * 100 * t) * 0.3
        data_int = (data * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, data_int)
        
        # Verify
        self.assertTrue(filepath.exists())
        
        # Read back and check
        sr, audio = wavfile.read(filepath)
        self.assertEqual(sr, sample_rate)
        self.assertEqual(len(audio), num_samples)
        self.assertTrue(audio.dtype == np.int16)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up temporary directories."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_dummy_wav(self, filepath: Path, duration: float = 1.0, sample_rate: int = 4000):
        """Create a dummy WAV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        data = np.sin(2 * np.pi * 60 * t) * 0.5
        data_int = (data * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, data_int)
    
    def test_custom_dataset_full_pipeline(self):
        """Test full pipeline for custom dataset."""
        # Setup input structure
        input_dir = Path(self.test_dir) / "input"
        output_dir = Path(self.test_dir) / "output"
        
        normal_dir = input_dir / "Normal"
        abnormal_dir = input_dir / "Abnormal"
        normal_dir.mkdir(parents=True)
        abnormal_dir.mkdir(parents=True)
        
        # Create multiple subjects with multiple recordings
        subjects = [
            ("normal", "0001", "male", "25", ["A", "M", "P"]),
            ("normal", "0002", "female", "30", ["A", "M"]),
            ("asd", "0001", "female", "5", ["A", "M", "P", "T"]),
        ]
        
        for condition, case_id, gender, age, locations in subjects:
            base_dir = normal_dir if condition == "normal" else abnormal_dir
            for loc in locations:
                filename = f"{condition}_case{case_id}_{gender}_{age}_10s_USA_{loc}.wav"
                self.create_dummy_wav(base_dir / filename)
        
        # Run organization
        stats = custom_organize(
            [normal_dir, abnormal_dir],
            output_dir,
            copy_files=True
        )
        
        # Verify results
        self.assertEqual(stats['num_subjects'], 3)
        self.assertEqual(stats['num_recordings'], 9)
        
        # Check metadata
        metadata_file = output_dir / "custom_metadata.csv"
        self.assertTrue(metadata_file.exists())
        
        df = pd.read_csv(metadata_file)
        self.assertEqual(len(df), 9)
        
        # Verify each subject
        for condition, case_id, gender, age, locations in subjects:
            subject_id = f"{condition}_case{case_id}"
            subject_dir = output_dir / f"subject_{subject_id}"
            self.assertTrue(subject_dir.exists())
            
            for loc in locations:
                rec_file = subject_dir / f"rec_{loc}.wav"
                self.assertTrue(rec_file.exists())


def run_tests(verbosity=1, test_download=False):
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFilenameParser))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetOrganization))
    suite.addTests(loader.loadTestsFromTestCase(TestWAVFileValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test data preparation scripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--test-download', action='store_true', 
                       help='Include download tests (slow, requires internet)')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Data Preparation Scripts - Test Suite")
    print("="*60)
    print()
    
    # Run tests
    success = run_tests(verbosity=2 if args.verbose else 1, test_download=args.test_download)
    
    print()
    print("="*60)
    if success:
        print("All tests passed! ✓")
        print("="*60)
        return 0
    else:
        print("Some tests failed! ✗")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
