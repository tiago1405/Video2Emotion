import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
from utils.face_matching import (
    Face,
    same_face,
    check_faces,
    create_face_embedding,
    annotate_face_emotion
)


class TestFaceClass(unittest.TestCase):
    """Test Face class functionality."""
    
    def test_face_initialization(self):
        """Test Face object creation."""
        embedding = [0.1] * 512
        location = {'x': 10, 'y': 20, 'w': 50, 'h': 60}
        
        face = Face(embedding=embedding, face_id=1, location=location)
        
        self.assertEqual(face.id, 1)
        self.assertEqual(len(face.locations), 1)
        self.assertEqual(face.locations[0], location)
        self.assertEqual(len(face.emotions), 0)
        self.assertEqual(len(face.dominant_emotions), 0)
    
    def test_add_location(self):
        """Test adding locations to Face."""
        face = Face(embedding=[0.1] * 512, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        
        new_location = {'x': 20, 'y': 30, 'w': 50, 'h': 60}
        face.add_location(new_location)
        
        self.assertEqual(len(face.locations), 2)
        self.assertEqual(face.locations[1], new_location)
    
    def test_add_emotion(self):
        """Test adding emotions to Face."""
        face = Face(embedding=[0.1] * 512, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        
        emotion_data = {'happy': 0.8, 'sad': 0.2}
        face.add_emotion(emotion_data)
        
        self.assertEqual(len(face.emotions), 1)
        self.assertEqual(face.emotions[0], emotion_data)


class TestSameFace(unittest.TestCase):
    """Test face matching logic."""
    
    @staticmethod
    def create_random_embedding(seed=None):
        """Create a realistic normalized embedding vector."""
        if seed is not None:
            np.random.seed(seed)
        # Create a random vector and normalize it
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def test_identical_embeddings(self):
        """Test that identical embeddings match."""
        embed = self.create_random_embedding(seed=42)
        
        # Test with different models
        for model in ["Facenet512", "VGG-Face", "ArcFace"]:
            result = same_face(embed, embed, model=model)
            self.assertTrue(result, f"Identical embeddings should match for {model}")
    
    def test_similar_embeddings(self):
        """Test similar embeddings (small distance)."""
        # Create a base embedding and a slightly perturbed version
        embed1 = self.create_random_embedding(seed=42)
        embed2 = np.array(embed1) + np.random.RandomState(43).randn(512) * 0.01
        embed2 = embed2 / np.linalg.norm(embed2)
        embed2 = embed2.tolist()
        
        result = same_face(embed1, embed2, model="Facenet512")
        self.assertTrue(result, "Similar embeddings should match")
    
    def test_different_embeddings(self):
        """Test very different embeddings."""
        # Create two completely different random embeddings
        embed1 = self.create_random_embedding(seed=42)
        embed2 = self.create_random_embedding(seed=100)
        
        result = same_face(embed1, embed2, model="Facenet512")
        self.assertFalse(result, "Different embeddings should not match")
    
    def test_unsupported_model(self):
        """Test handling of unsupported model."""
        embed1 = self.create_random_embedding(seed=42)
        embed2 = self.create_random_embedding(seed=43)
        
        with self.assertRaises(ValueError):
            same_face(embed1, embed2, model="UnsupportedModel")


class TestCheckFaces(unittest.TestCase):
    """Test checking embeddings against known faces."""
    
    @staticmethod
    def create_random_embedding(seed=None):
        """Create a realistic normalized embedding vector."""
        if seed is not None:
            np.random.seed(seed)
        # Create a random vector and normalize it
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def test_matching_face_found(self):
        """Test finding a matching face."""
        embed1 = self.create_random_embedding(seed=42)
        face1 = Face(embedding=embed1, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        
        faces = {tuple(embed1): face1}
        
        # Check with same embedding
        result = check_faces(embed1, faces, model="Facenet512")
        
        self.assertIsNotNone(result, "Should find matching face")
        self.assertEqual(list(result), embed1)
    
    def test_no_matching_face(self):
        """Test when no matching face exists."""
        embed1 = self.create_random_embedding(seed=42)
        embed2 = self.create_random_embedding(seed=100)
        
        face1 = Face(embedding=embed1, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        faces = {tuple(embed1): face1}
        
        result = check_faces(embed2, faces, model="Facenet512")
        
        self.assertIsNone(result, "Should not find matching face")
    
    def test_empty_faces_dict(self):
        """Test with empty faces dictionary."""
        embed = self.create_random_embedding(seed=42)
        faces = {}
        
        result = check_faces(embed, faces, model="Facenet512")
        
        self.assertIsNone(result, "Should return None for empty faces dict")
    
    def test_multiple_faces(self):
        """Test checking against multiple faces."""
        embed1 = self.create_random_embedding(seed=42)
        embed2 = self.create_random_embedding(seed=43)
        embed3 = self.create_random_embedding(seed=100)
        
        face1 = Face(embedding=embed1, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        face2 = Face(embedding=embed2, face_id=2, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        
        faces = {tuple(embed1): face1, tuple(embed2): face2}
        
        # Should match embed2
        result = check_faces(embed2, faces, model="Facenet512")
        self.assertIsNotNone(result)
        
        # Should not match embed3
        result = check_faces(embed3, faces, model="Facenet512")
        self.assertIsNone(result)


class TestAnnotateFaceEmotion(unittest.TestCase):
    """Test face emotion annotation."""
    
    def setUp(self):
        """Create temporary directory and test image."""
        self.temp_dir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.temp_dir, "test.jpg")
        
        # Create simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [100, 150, 200]
        cv2.imwrite(self.img_path, img)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_annotation_without_face_locations(self):
        """Test annotation without face ID tracking."""
        emotional_analysis = [{
            'region': {'x': 10, 'y': 10, 'w': 30, 'h': 30},
            'dominant_emotion': 'happy',
            'emotion': {'happy': 0.9, 'sad': 0.1}
        }]
        
        result = annotate_face_emotion(
            img_path=self.img_path,
            emotional_analysis=emotional_analysis
        )
        
        self.assertIsNotNone(result, "Should return annotated image")
        self.assertEqual(result.shape, (100, 100, 3), "Should maintain image dimensions")
    
    def test_annotation_with_face_locations(self):
        """Test annotation with face ID tracking."""
        region = {'x': 10, 'y': 10, 'w': 30, 'h': 30}
        emotional_analysis = [{
            'region': region,
            'dominant_emotion': 'sad',
            'emotion': {'happy': 0.2, 'sad': 0.8}
        }]
        
        face_locations = [(1, region)]
        
        result = annotate_face_emotion(
            img_path=self.img_path,
            emotional_analysis=emotional_analysis,
            face_locations=face_locations
        )
        
        self.assertIsNotNone(result)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        emotional_analysis = [{
            'region': {'x': 10, 'y': 10, 'w': 30, 'h': 30},
            'dominant_emotion': 'happy'
        }]
        
        with self.assertRaises(FileNotFoundError):
            annotate_face_emotion(
                img_path="/invalid/path.jpg",
                emotional_analysis=emotional_analysis
            )
    
    def test_none_image_path(self):
        """Test handling of None image path."""
        with self.assertRaises(ValueError):
            annotate_face_emotion(
                img_path=None,
                emotional_analysis=[{}]
            )
    
    def test_empty_emotional_analysis(self):
        """Test handling of empty emotional analysis."""
        with self.assertRaises(ValueError):
            annotate_face_emotion(
                img_path=self.img_path,
                emotional_analysis=[]
            )
    
    def test_multiple_faces_annotation(self):
        """Test annotating multiple faces."""
        emotional_analysis = [
            {
                'region': {'x': 10, 'y': 10, 'w': 20, 'h': 20},
                'dominant_emotion': 'happy',
                'emotion': {'happy': 0.9}
            },
            {
                'region': {'x': 50, 'y': 50, 'w': 20, 'h': 20},
                'dominant_emotion': 'sad',
                'emotion': {'sad': 0.8}
            }
        ]
        
        result = annotate_face_emotion(
            img_path=self.img_path,
            emotional_analysis=emotional_analysis
        )
        
        self.assertIsNotNone(result, "Should handle multiple faces")


class TestFaceMatchingStress(unittest.TestCase):
    """Stress tests for face matching."""
    
    @staticmethod
    def create_random_embedding(seed=None):
        """Create a realistic normalized embedding vector."""
        if seed is not None:
            np.random.seed(seed)
        # Create a random vector and normalize it
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def test_many_faces_check(self):
        """Stress test: check against many known faces."""
        # Create 100 different face embeddings
        faces = {}
        for i in range(100):
            embed = self.create_random_embedding(seed=i)
            face = Face(embedding=embed, face_id=i, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
            faces[tuple(embed)] = face
        
        # Check a new embedding
        test_embed = self.create_random_embedding(seed=50)
        result = check_faces(test_embed, faces, model="Facenet512")
        
        # Should find the matching face (id=50)
        self.assertIsNotNone(result)
    
    def test_face_location_tracking(self):
        """Stress test: track face across many frames."""
        embed = self.create_random_embedding(seed=42)
        face = Face(embedding=embed, face_id=1, location={'x': 0, 'y': 0, 'w': 10, 'h': 10})
        
        # Add 500 locations (simulating 500 frames)
        for i in range(500):
            face.add_location({'x': i, 'y': i, 'w': 10, 'h': 10})
        
        self.assertEqual(len(face.locations), 501, "Should track all locations")


if __name__ == '__main__':
    unittest.main()