import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
from utils.video_utils import video_to_images


class TestVideoToImages(unittest.TestCase):
    """Test video frame extraction functionality."""
    
    def setUp(self):
        """Create temporary directories and a test video."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test video (10 frames at 30fps)
        self._create_test_video(self.video_path, num_frames=10, fps=30)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_video(self, video_path: str, num_frames: int = 10, fps: int = 30):
        """Create a simple test video with colored frames."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (100, 100))
        
        for i in range(num_frames):
            # Create frame with changing color
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :] = [i * 25 % 256, 100, 150]
            out.write(frame)
        
        out.release()
    
    def test_video_extraction_by_frames(self):
        """Test extracting frames by frame count."""
        video_to_images(
            video_path=self.video_path,
            out_dir=self.output_dir,
            every_n=2.0,
            time_or_frames="frames"
        )
        
        # Check that frames were extracted
        frame_dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(frame_dirs), 1, "Should create one frame directory")
        
        frame_files = os.listdir(os.path.join(self.output_dir, frame_dirs[0]))
        self.assertGreater(len(frame_files), 0, "Should extract at least one frame")
        self.assertTrue(all(f.endswith('.jpg') for f in frame_files), "All frames should be JPG")
    
    def test_video_extraction_by_time(self):
        """Test extracting frames by time interval."""
        video_to_images(
            video_path=self.video_path,
            out_dir=self.output_dir,
            every_n=1.0,
            time_or_frames="time"
        )
        
        frame_dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertGreater(len(frame_dirs), 0, "Should create frame directories")
    
    def test_invalid_video_path(self):
        """Test handling of non-existent video."""
        invalid_path = os.path.join(self.temp_dir, "nonexistent.mp4")
        
        # Should handle gracefully without raising exception
        result = video_to_images(
            video_path=invalid_path,
            out_dir=self.output_dir,
            every_n=1.0
        )
        
        self.assertIsNone(result, "Should return None for invalid video")
    
    def test_output_directory_creation(self):
        """Test that output directories are created."""
        video_to_images(
            video_path=self.video_path,
            out_dir=self.output_dir,
            every_n=1.0
        )
        
        # Check that a subdirectory was created
        subdirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertGreater(len(subdirs), 0, "Should create subdirectory for video frames")


class TestVideoExtractionStress(unittest.TestCase):
    """Stress tests for video extraction."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_large_frame_extraction(self):
        """Stress test: extract from 100-frame video."""
        video_path = os.path.join(self.temp_dir, "large_video.mp4")
        
        # Create 100-frame video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))
        
        for i in range(100):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Extract every 10th frame
        video_to_images(
            video_path=video_path,
            out_dir=self.output_dir,
            every_n=10.0,
            time_or_frames="frames"
        )
        
        frame_dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertGreater(len(frame_dirs), 0, "Should handle large videos")
    
    def test_multiple_consecutive_extractions(self):
        """Stress test: multiple extractions in sequence."""
        for i in range(3):
            video_path = os.path.join(self.temp_dir, f"video_{i}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30, (100, 100))
            
            for j in range(10):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            video_to_images(
                video_path=video_path,
                out_dir=self.output_dir,
                every_n=1.0
            )
        
        frame_dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(frame_dirs), 3, "Should handle multiple sequential extractions")


if __name__ == '__main__':
    unittest.main()