# tests/test_statistics.py
import unittest
import pandas as pd
import tempfile
import shutil
import os
import json
from utils.statistics_utils import (
    calculate_emotion_distribution,
    calculate_temporal_patterns,
    calculate_emotion_transitions,
    compare_videos,
    calculate_face_statistics,
    generate_summary_report,
    generate_statistics_csv,
    print_statistics_summary
)


class TestEmotionDistribution(unittest.TestCase):
    """Test emotion distribution calculations."""
    
    def test_basic_distribution(self):
        """Test basic emotion distribution."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 3 + ['neutral'] * 2
        })
        
        result = calculate_emotion_distribution(df)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['emotion'], 'happy')
        self.assertEqual(result.iloc[0]['count'], 5)
        self.assertAlmostEqual(result.iloc[0]['percentage'], 50.0, places=1)
    
    def test_distribution_with_grouping(self):
        """Test emotion distribution grouped by video."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5,
            'source_directory': ['video1'] * 5 + ['video2'] * 5
        })
        
        result = calculate_emotion_distribution(df, group_by='source_directory')
        
        self.assertGreater(len(result), 0)
        self.assertIn('source_directory', result.columns)
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_emotion_distribution(df)
        
        self.assertTrue(result.empty)


class TestTemporalPatterns(unittest.TestCase):
    """Test temporal pattern analysis."""
    
    def test_basic_temporal_analysis(self):
        """Test basic temporal pattern calculation."""
        df = pd.DataFrame({
            'frame_id': range(20),
            'dominant_emotion': ['happy'] * 10 + ['sad'] * 10
        })
        
        result = calculate_temporal_patterns(df, window_size=5)
        
        self.assertEqual(len(result), 20)
        self.assertIn('emotion_stability', result.columns)
        self.assertIn('emotion_duration', result.columns)
        self.assertIn('emotion_changed', result.columns)
    
    def test_emotion_stability(self):
        """Test emotion stability calculation."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 10  # No changes
        })
        
        result = calculate_temporal_patterns(df, window_size=5)
        
        # High stability (no changes)
        self.assertGreater(result['emotion_stability'].mean(), 0.9)
    
    def test_emotion_duration(self):
        """Test emotion duration tracking."""
        df = pd.DataFrame({
            'frame_id': range(15),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5 + ['neutral'] * 5
        })
        
        result = calculate_temporal_patterns(df, window_size=3)
        
        # Check that duration resets on emotion change
        self.assertEqual(result.iloc[0]['emotion_duration'], 1)
        self.assertEqual(result.iloc[4]['emotion_duration'], 5)
        self.assertEqual(result.iloc[5]['emotion_duration'], 1)


class TestEmotionTransitions(unittest.TestCase):
    """Test emotion transition analysis."""
    
    def test_basic_transitions(self):
        """Test basic transition calculation."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy', 'happy', 'sad', 'sad', 'sad', 
                               'neutral', 'neutral', 'happy', 'happy', 'sad']
        })
        
        result = calculate_emotion_transitions(df)
        
        self.assertGreater(len(result), 0)
        self.assertIn('dominant_emotion', result.columns)
        self.assertIn('next_emotion', result.columns)
        self.assertIn('probability', result.columns)
    
    def test_transition_probabilities(self):
        """Test that probabilities sum to 1 for each emotion."""
        df = pd.DataFrame({
            'frame_id': range(100),
            'dominant_emotion': (['happy'] * 50 + ['sad'] * 50)
        })
        
        result = calculate_emotion_transitions(df)
        
        # Check probabilities
        for emotion in result['dominant_emotion'].unique():
            emotion_probs = result[result['dominant_emotion'] == emotion]['probability'].sum()
            self.assertAlmostEqual(emotion_probs, 1.0, places=5)


class TestVideoComparison(unittest.TestCase):
    """Test video comparison functionality."""
    
    def test_compare_two_videos(self):
        """Test comparing two videos."""
        df = pd.DataFrame({
            'frame_id': range(20),
            'dominant_emotion': ['happy'] * 10 + ['sad'] * 10,
            'source_directory': ['video1'] * 10 + ['video2'] * 10
        })
        
        result = compare_videos(df)
        
        self.assertEqual(len(result), 2)
        self.assertIn('video', result.columns)
        self.assertIn('total_frames', result.columns)
        self.assertIn('most_common_emotion', result.columns)
    
    def test_emotion_percentages_in_comparison(self):
        """Test that emotion percentages are calculated."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5,
            'source_directory': ['video1'] * 10
        })
        
        result = compare_videos(df)
        
        self.assertIn('happy_pct', result.columns)
        self.assertIn('sad_pct', result.columns)
        self.assertAlmostEqual(result.iloc[0]['happy_pct'], 50.0, places=1)


class TestFaceStatistics(unittest.TestCase):
    """Test face-level statistics."""
    
    def test_basic_face_stats(self):
        """Test basic face statistics calculation."""
        df = pd.DataFrame({
            'frame_id': range(20),
            'face_id': [1] * 10 + [2] * 10,
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5 + ['neutral'] * 10
        })
        
        result = calculate_face_statistics(df)
        
        self.assertEqual(len(result), 2)
        self.assertIn('face_id', result.columns)
        self.assertIn('total_appearances', result.columns)
        self.assertIn('dominant_emotion', result.columns)
    
    def test_emotion_variety_per_face(self):
        """Test emotion variety calculation per face."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'face_id': [1] * 10,
            'dominant_emotion': ['happy', 'sad', 'neutral', 'angry', 'fearful'] * 2
        })
        
        result = calculate_face_statistics(df)
        
        self.assertEqual(result.iloc[0]['emotion_variety'], 5)


class TestSummaryReport(unittest.TestCase):
    """Test summary report generation."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_report(self):
        """Test basic report generation."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5
        })
        
        report = generate_summary_report(df)
        
        self.assertIn('overview', report)
        self.assertIn('emotion_distribution', report)
        self.assertEqual(report['overview']['total_frames'], 10)
    
    def test_report_with_video_data(self):
        """Test report with video comparison."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 5 + ['sad'] * 5,
            'source_directory': ['video1'] * 10
        })
        
        report = generate_summary_report(df)
        
        self.assertIn('video_comparison', report)
    
    def test_report_save_to_file(self):
        """Test saving report to JSON file."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 10
        })
        
        output_path = os.path.join(self.temp_dir, "report.json")
        report = generate_summary_report(df, output_path=output_path)
        
        self.assertTrue(os.path.exists(output_path))
        
        # Verify JSON is valid
        with open(output_path, 'r') as f:
            loaded_report = json.load(f)
        
        self.assertEqual(loaded_report['overview']['total_frames'], 10)


class TestStatisticsCSVGeneration(unittest.TestCase):
    """Test CSV statistics generation."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_multiple_csvs(self):
        """Test generating multiple CSV files."""
        df = pd.DataFrame({
            'frame_id': range(20),
            'dominant_emotion': ['happy'] * 10 + ['sad'] * 10,
            'source_directory': ['video1'] * 10 + ['video2'] * 10
        })
        
        files = generate_statistics_csv(df, self.temp_dir, prefix="test")
        
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(os.path.exists(f))
            self.assertTrue(f.endswith('.csv'))
    
    def test_csv_files_readable(self):
        """Test that generated CSV files are readable."""
        df = pd.DataFrame({
            'frame_id': range(10),
            'dominant_emotion': ['happy'] * 10
        })
        
        files = generate_statistics_csv(df, self.temp_dir)
        
        for f in files:
            df_read = pd.read_csv(f)
            self.assertGreater(len(df_read), 0)


class TestStressStatistics(unittest.TestCase):
    """Stress tests for statistics generation."""
    
    def test_large_dataset_statistics(self):
        """Stress test: statistics on large dataset."""
        # Simulate 500 videos with 100 frames each
        df = pd.DataFrame({
            'frame_id': list(range(100)) * 500,
            'dominant_emotion': (['happy', 'sad', 'neutral', 'angry', 'fearful'] * 20) * 500,
            'source_directory': [f'video_{i//100}' for i in range(50000)]
        })
        
        report = generate_summary_report(df)
        
        self.assertEqual(report['overview']['total_frames'], 50000)
        self.assertIn('video_comparison', report)
    
    def test_many_emotion_transitions(self):
        """Stress test: many emotion transitions."""
        emotions = ['happy', 'sad', 'neutral', 'angry', 'fearful', 'disgust', 'surprise']
        df = pd.DataFrame({
            'frame_id': range(1000),
            'dominant_emotion': [emotions[i % len(emotions)] for i in range(1000)]
        })
        
        result = calculate_emotion_transitions(df)
        
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()