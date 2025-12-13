# tests/test_system_utils.py
import unittest
import os
import tempfile
import shutil
from utils.system_utils import crawl_dir


class TestCrawlDir(unittest.TestCase):
    """Test directory crawling functionality."""
    
    def setUp(self):
        """Create temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_crawl_mp4_files(self):
        """Test finding MP4 files."""
        # Create test files
        open(os.path.join(self.temp_dir, "video1.mp4"), 'w').close()
        open(os.path.join(self.temp_dir, "video2.mp4"), 'w').close()
        open(os.path.join(self.temp_dir, "image.jpg"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 2, "Should find exactly 2 MP4 files")
        self.assertTrue(all(f.endswith('.mp4') for f in files), "All files should be MP4")
    
    def test_crawl_with_dot_extension(self):
        """Test crawl_dir with dot in extension."""
        open(os.path.join(self.temp_dir, "video.mp4"), 'w').close()
        
        files = crawl_dir(self.temp_dir, ".mp4")
        
        self.assertEqual(len(files), 1, "Should handle extension with dot")
    
    def test_empty_directory(self):
        """Test crawling empty directory."""
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 0, "Should return empty list for empty directory")
    
    def test_no_matching_files(self):
        """Test when no files match extension."""
        open(os.path.join(self.temp_dir, "image.jpg"), 'w').close()
        open(os.path.join(self.temp_dir, "photo.png"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 0, "Should return empty list when no files match")
    
    def test_full_paths_returned(self):
        """Test that full paths are returned."""
        open(os.path.join(self.temp_dir, "video.mp4"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 1)
        self.assertTrue(os.path.isabs(files[0]), "Should return absolute paths")
        self.assertTrue(files[0].endswith("video.mp4"), "Path should contain filename")
    
    def test_case_insensitive_extension(self):
        """Test that extension matching is case-sensitive (as implemented)."""
        open(os.path.join(self.temp_dir, "video.MP4"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        # Current implementation is case-sensitive
        self.assertEqual(len(files), 0, "Case-sensitive matching")


class TestCrawlDirStress(unittest.TestCase):
    """Stress tests for directory crawling."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_large_directory(self):
        """Stress test: crawl directory with many files."""
        # Create 500 files
        for i in range(500):
            if i % 5 == 0:
                open(os.path.join(self.temp_dir, f"video_{i}.mp4"), 'w').close()
            else:
                open(os.path.join(self.temp_dir, f"file_{i}.txt"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 100, "Should handle large directories")
    
    def test_subdirectories_ignored(self):
        """Test that subdirectories are properly ignored."""
        os.makedirs(os.path.join(self.temp_dir, "subdir"))
        open(os.path.join(self.temp_dir, "video.mp4"), 'w').close()
        open(os.path.join(self.temp_dir, "subdir", "video.mp4"), 'w').close()
        
        files = crawl_dir(self.temp_dir, "mp4")
        
        self.assertEqual(len(files), 1, "Should not include files from subdirectories")


if __name__ == '__main__':
    unittest.main()