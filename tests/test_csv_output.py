# tests/test_csv_output.py
import unittest
import os
import tempfile
import shutil
import pandas as pd
from io import StringIO
import sys


class TestCSVOutput(unittest.TestCase):
    """Test CSV output functionality."""
    
    def setUp(self):
        """Create temporary output directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_creation(self):
        """Test that CSV files are created successfully."""
        # Create sample emotion data
        emotion_data = {
            'frame_id': [1, 2, 3],
            'dominant_emotion': ['happy', 'sad', 'neutral'],
            'emotions': [
                {'happy': 0.9, 'sad': 0.1},
                {'happy': 0.2, 'sad': 0.8},
                {'happy': 0.5, 'sad': 0.5}
            ]
        }
        
        df = pd.DataFrame(emotion_data)
        output_path = os.path.join(self.output_dir, "test_output.csv")
        df.to_csv(output_path, index=False)
        
        self.assertTrue(os.path.exists(output_path), "CSV file should be created")
    
    def test_csv_read_back(self):
        """Test that CSV can be read back correctly."""
        emotion_data = {
            'frame_id': [1, 2, 3],
            'dominant_emotion': ['happy', 'sad', 'neutral'],
            'source_directory': ['video1', 'video1', 'video1']
        }
        
        df = pd.DataFrame(emotion_data)
        output_path = os.path.join(self.output_dir, "test.csv")
        df.to_csv(output_path, index=False)
        
        # Read back
        df_read = pd.read_csv(output_path)
        
        self.assertEqual(len(df_read), 3, "Should read correct number of rows")
        self.assertListEqual(list(df_read.columns), ['frame_id', 'dominant_emotion', 'source_directory'])
    
    def test_combined_csv_concat(self):
        """Test concatenating multiple CSVs."""
        df1 = pd.DataFrame({
            'frame_id': [1, 2],
            'dominant_emotion': ['happy', 'sad'],
            'source_directory': ['video1', 'video1']
        })
        
        df2 = pd.DataFrame({
            'frame_id': [1, 2],
            'dominant_emotion': ['neutral', 'angry'],
            'source_directory': ['video2', 'video2']
        })
        
        df_combined = pd.concat([df1, df2], ignore_index=True)
        output_path = os.path.join(self.output_dir, "combined.csv")
        df_combined.to_csv(output_path, index=False)
        
        df_read = pd.read_csv(output_path)
        
        self.assertEqual(len(df_read), 4, "Combined CSV should have 4 rows")
        self.assertEqual(len(df_read['source_directory'].unique()), 2, "Should have 2 unique videos")
    
    def test_csv_with_index(self):
        """Test CSV output with index."""
        df = pd.DataFrame({
            'frame_id': [1, 2, 3],
            'dominant_emotion': ['happy', 'sad', 'neutral']
        })
        
        output_path = os.path.join(self.output_dir, "with_index.csv")
        df.to_csv(output_path, index=True)
        
        df_read = pd.read_csv(output_path)
        
        # Should have an 'Unnamed: 0' column from the index
        self.assertIn('Unnamed: 0', df_read.columns, "Index should be included")
    
    def test_csv_without_index(self):
        """Test CSV output without index."""
        df = pd.DataFrame({
            'frame_id': [1, 2, 3],
            'dominant_emotion': ['happy', 'sad', 'neutral']
        })
        
        output_path = os.path.join(self.output_dir, "no_index.csv")
        df.to_csv(output_path, index=False)
        
        df_read = pd.read_csv(output_path)
        
        self.assertEqual(len(df_read.columns), 2, "Should have 2 columns (no index)")


class TestCSVStress(unittest.TestCase):
    """Stress tests for CSV output."""
    
    def setUp(self):
        """Create temporary output directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_large_dataframe_export(self):
        """Stress test: export large DataFrame."""
        # Create DataFrame with 10,000 rows (simulating 500 videos analysis)
        data = {
            'frame_id': range(10000),
            'dominant_emotion': ['happy', 'sad', 'neutral', 'angry', 'fearful'] * 2000,
            'source_directory': [f'video_{i//50}' for i in range(10000)]
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, "large_output.csv")
        df.to_csv(output_path, index=False)
        
        self.assertTrue(os.path.exists(output_path), "Large CSV should be created")
        
        df_read = pd.read_csv(output_path)
        self.assertEqual(len(df_read), 10000, "Should handle large DataFrames")
    
    def test_multiple_csv_concat(self):
        """Stress test: concatenate many CSVs."""
        dfs = []
        for i in range(100):
            df = pd.DataFrame({
                'frame_id': range(100),
                'dominant_emotion': ['happy'] * 100,
                'source_directory': [f'video_{i}'] * 100
            })
            dfs.append(df)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        
        self.assertEqual(len(df_combined), 10000, "Should concatenate multiple DataFrames")
        self.assertEqual(len(df_combined['source_directory'].unique()), 100, "Should track 100 videos")


if __name__ == '__main__':
    unittest.main()