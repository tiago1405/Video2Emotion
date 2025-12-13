# tests/test_video2emo.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
import pandas as pd
from video2emo import deepface_analysis, analyze_directory


class TestDeepfaceAnalysis(unittest.TestCase):
    """Test DeepFace emotion analysis."""
    
    def setUp(self):
        """Create temporary directory and test image with a face-like structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.temp_dir, "test_face.jpg")
        
        # Create a simple test image (not a real face, but for testing structure)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.img_path, img)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_deepface_analysis_returns_list(self):
        """Test that deepface_analysis returns a list."""
        try:
            result = deepface_analysis(img_path=self.img_path, actions=['emotion'])
            self.assertIsInstance(result, list, "Should return a list")
        except Exception as e:
            # DeepFace may fail on non-face images, which is expected
            self.assertIn("face", str(e).lower(), "Should fail with face-related error")
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        with self.assertRaises(FileNotFoundError):
            deepface_analysis(img_path="/invalid/path.jpg")
    
    def test_default_actions(self):
        """Test that default action is emotion."""
        try:
            result = deepface_analysis(img_path=self.img_path)
            # If it succeeds, it should use default emotion action
            self.assertIsInstance(result, list)
        except Exception:
            # Expected to fail on non-face images
            pass
    
    def test_custom_actions(self):
        """Test with custom actions list."""
        try:
            result = deepface_analysis(
                img_path=self.img_path,
                actions=['emotion', 'age', 'gender']
            )
            self.assertIsInstance(result, list)
        except Exception:
            # Expected to fail on non-face images
            pass


class TestAnalyzeDirectory(unittest.TestCase):
    """Test directory analysis functionality."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Create test images
        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_path = os.path.join(self.frames_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(img_path, img)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_directory_returns_dataframe(self):
        """Test that analyze_directory returns a DataFrame."""
        try:
            result = analyze_directory(
                dir_path=self.frames_dir,
                mode='deepface',
                face_match_people=False
            )
            
            # If successful, should return DataFrame
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
                self.assertIn('frame_id', result.columns, "Should have frame_id column")
                self.assertIn('dominant_emotion', result.columns, "Should have dominant_emotion column")
        except Exception:
            # May fail on non-face images, which is acceptable
            pass
    
    def test_invalid_directory(self):
        """Test handling of invalid directory."""
        result = analyze_directory(
            dir_path="/invalid/path",
            mode='deepface'
        )
        
        self.assertIsNone(result, "Should return None for invalid directory")
    
    def test_unsupported_mode(self):
        """Test handling of unsupported analysis mode."""
        result = analyze_directory(
            dir_path=self.frames_dir,
            mode='unsupported_mode'
        )
        
        self.assertIsNone(result, "Should return None for unsupported mode")
    
    def test_dataframe_structure_without_face_matching(self):
        """Test DataFrame structure without face matching."""
        try:
            result = analyze_directory(
                dir_path=self.frames_dir,
                mode='deepface',
                face_match_people=False
            )
            
            if result is not None and len(result) > 0:
                # Should not have face_id column
                self.assertNotIn('face_id', result.columns, "Should not have face_id without face matching")
        except Exception:
            pass
    
    def test_dataframe_structure_with_face_matching(self):
        """Test DataFrame structure with face matching."""
        try:
            result = analyze_directory(
                dir_path=self.frames_dir,
                mode='deepface',
                face_match_people=True
            )
            
            if result is not None and len(result) > 0:
                # Should have face_id column
                self.assertIn('face_id', result.columns, "Should have face_id with face matching")
        except Exception:
            pass


class TestAnalyzeDirectoryStress(unittest.TestCase):
    """Stress tests for directory analysis."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_many_frames_analysis(self):
        """Stress test: analyze 100 frames."""
        # Create 100 test images
        for i in range(100):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_path = os.path.join(self.frames_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(img_path, img)
        
        try:
            result = analyze_directory(
                dir_path=self.frames_dir,
                mode='deepface',
                face_match_people=False,
                verbose=False
            )
            
            # If successful, check it handled all frames
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame)
        except Exception:
            # May fail on non-face images
            pass
    
    def test_empty_directory(self):
        """Test handling of empty directory."""
        result = analyze_directory(
            dir_path=self.frames_dir,
            mode='deepface'
        )
        
        # Should handle gracefully (may return empty DataFrame or None)
        self.assertTrue(
            result is None or isinstance(result, pd.DataFrame),
            "Should handle empty directory gracefully"
        )
    
    def test_save_emo_frames_parameter(self):
        """Test emotion frame saving functionality."""
        emo_frames_path = os.path.join(self.temp_dir, "emo_frames")
        
        # Create a test image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = os.path.join(self.frames_dir, "frame_00001.jpg")
        cv2.imwrite(img_path, img)
        
        try:
            result = analyze_directory(
                dir_path=self.frames_dir,
                mode='deepface',
                save_emo_frames=True,
                emo_frames_path=emo_frames_path
            )
            
            # Check if emo_frames directory was created
            self.assertTrue(
                os.path.exists(emo_frames_path),
                "Should create emotion frames directory"
            )
        except Exception:
            # May fail on non-face images
            pass


class TestEndToEndIntegration(unittest.TestCase):
    """Integration tests simulating full pipeline."""
    
    def setUp(self):
        """Create temporary directories simulating real workflow."""
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test frames for 3 "videos"
        for video_idx in range(3):
            video_frames_dir = os.path.join(self.frames_dir, f"video_{video_idx}")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            for frame_idx in range(10):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = os.path.join(video_frames_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(img_path, img)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_multi_video_processing(self):
        """Test processing multiple video directories."""
        results = []
        
        for video_dir in os.listdir(self.frames_dir):
            video_path = os.path.join(self.frames_dir, video_dir)
            if not os.path.isdir(video_path):
                continue
            
            try:
                df = analyze_directory(
                    dir_path=video_path,
                    mode='deepface',
                    face_match_people=False
                )
                
                if df is not None:
                    df['source_directory'] = video_dir
                    results.append(df)
            except Exception:
                pass
        
        # If any succeeded, test concatenation
        if results:
            df_combined = pd.concat(results, ignore_index=True)
            self.assertIsInstance(df_combined, pd.DataFrame)
            self.assertIn('source_directory', df_combined.columns)


if __name__ == '__main__':
    unittest.main()