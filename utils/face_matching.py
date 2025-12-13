from deepface import DeepFace
from deepface.modules.verification import find_distance, l2_normalize, find_threshold
import pandas as pd
import numpy as np
import cv2
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class Face:
    """A tracked face across multiple frames.
    
    Attributes:
        embedding: Face embedding vector
        id: Unique identifier for this face
        locations: List of facial area dictionaries across frames
        emotions: List of emotion data for this face
        dominant_emotions: List of dominant emotions detected
        frame_paths: List of frame paths where this face appears
    """
    
    def __init__(self, embedding: list, face_id: int, location: dict):
        self.embedding = embedding
        self.id = face_id
        self.locations = [location]
        self.emotions = []
        self.dominant_emotions = []
        self.frame_paths = []
    
    def add_location(self, facial_area: dict) -> None:
        """Add a new location where this face was detected."""
        self.locations.append(facial_area)
    
    def add_emotion(self, emotion_data: dict) -> None:
        """Add emotion data for this face."""
        self.emotions.append(emotion_data)

def same_face(embed1: list, embed2: list, model: str) -> bool:
    """Return True if two embeddings belong to the same face.
    
    Chooses DeepFace's recommended distance metric and threshold for each model
    using `find_threshold`.
    
    Args:
        embed1: First face embedding vector
        embed2: Second face embedding vector
        model: Name of the face recognition model
    
    Returns:
        True if embeddings match the same face, False otherwise
    """
    match model:
        case "VGG-Face" | "DeepFace" | "ArcFace" | "SFace":
            distance_metric = "cosine"
            dist = find_distance(embed1, embed2, distance_metric=distance_metric)
        case "Facenet" | "Facenet512" | "OpenFace" | "Dlib":
            distance_metric = "euclidean_l2"
            dist = find_distance(
                l2_normalize(embed1),
                l2_normalize(embed2),
                distance_metric=distance_metric
            )
        case "DeepID":
            distance_metric = "euclidean"
            dist = find_distance(embed1, embed2, distance_metric=distance_metric)
        case _:
            raise ValueError(f"Unsupported model: {model}")

    threshold = find_threshold(model, distance_metric)
    return dist <= threshold

def check_faces(embed1: list, faces: dict, model: str) -> list | None:
    """Check if a face (embedding) matches any known faces.
    
    Args:
        embed1: Face embedding to check
        faces: Dictionary of known Face objects
        model: Name of the face recognition model
    
    Returns:
        Matching face embedding if found, None otherwise
    """
    for _, face in faces.items():
        if not isinstance(face, Face):
            print(f"Warning: Expected Face object, got {type(face).__name__}")
            continue
        
        if same_face(embed1=embed1, embed2=face.embedding, model=model):
            return face.embedding
    
    return None

def create_face_embedding(img_path: str, model: str = "Facenet512") -> list:
    """Create face embeddings for all faces in an image.
    
    Args:
        img_path: Path to the image file
        model: Name of the face recognition model
    
    Returns:
        List of embedding dictionaries for detected faces
    """
    embeddings = DeepFace.represent(img_path=img_path, model_name=model, enforce_detection=False)
    return embeddings

def find_frames_by_face(face_img_path: str, 
                        frames_path: str, 
                        model_name: str = "Facenet512") -> pd.DataFrame:
    """Find all frames containing a specific face.
    
    Args:
        face_img_path: Path to reference face image
        frames_path: Directory containing frames to search
        model_name: Name of the face recognition model
    
    Returns:
        DataFrame with matching frames and similarity scores
    """
    matches = DeepFace.find(
        img_path=face_img_path, 
        db_path=frames_path, 
        model_name=model_name, 
        enforce_detection=False
    )
    return matches[0] if matches else pd.DataFrame()

def annotate_face_emotion(img_path: str, 
                          emotional_analysis: list, 
                          face_locations: list = None, 
                          text_color: tuple = (0, 255, 0)) -> np.ndarray | None:
    """Annotate faces in an image with a box, emotion information, and face id.
    
    Args:
        img_path: Path to the image file
        emotional_analysis: List of emotion analysis results
        face_locations: List of (face_id, region) tuples for face identification
        text_color: RGB color tuple for text annotation (default: green)
    
    Returns:
        Annotated image as numpy array, or None if inputs are invalid
    """
    if img_path is None:
        raise ValueError("No path to image provided")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    if emotional_analysis is None or not emotional_analysis:
        raise ValueError("No emotional analysis provided")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    anno_img = img.copy()
    
    for analysis in emotional_analysis:
        region = analysis.get('region')
        if region is None:
            continue
        
        x1, y1 = region.get('x', 0), region.get('y', 0)
        dx, dy = region.get('w', 0), region.get('h', 0)
        x2 = x1 + dx
        y2 = y1 + dy
        
        # Draw rectangle around face
        cv2.rectangle(anno_img, (x1, y1), (x2, y2), text_color, 4)
        
        # Annotate face_id if available
        if face_locations is not None:
            try:
                matching_faces = [f[0] for f in face_locations if f[1] == region]
                if matching_faces:
                    face_id = matching_faces[0]
                    cv2.putText(anno_img, 
                               f"Face ID: {face_id}", 
                               org=(x1, y2 + 80), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1.25,
                               color=text_color,
                               thickness=3,
                               lineType=cv2.LINE_AA)
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not match face location: {e}")
        
        # Annotate the dominant emotion
        dominant_emotion = analysis.get('dominant_emotion', 'Unknown')
        cv2.putText(anno_img, 
                   f"Emotion: {dominant_emotion}",
                   org=(x1, y2 + 40), 
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.25,
                   color=text_color,
                   thickness=3,
                   lineType=cv2.LINE_AA)
    
    return anno_img