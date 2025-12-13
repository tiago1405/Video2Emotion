import os
import cv2
import pandas as pd
from deepface import DeepFace
import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from utils.system_utils import crawl_dir
from utils.face_matching import (create_face_embedding, 
                                 check_faces, 
                                 find_frames_by_face, 
                                 annotate_face_emotion,
                                 Face)


def deepface_analysis(img_path: str, 
                      actions: list = None,
                      save_emo_frames: bool = False,
                      emo_frames_path: str = None,
                      face_locations: list = None,
                      verbose: bool = False) -> list:
    """Analyze emotions in an image using DeepFace.
    
    Args:
        img_path: Path to the image file
        actions: List of analysis actions to perform (defaults
         to emotion)
    
    Returns:
        List of analysis results for detected faces
    """
    if actions is None:
        actions = ['emotion']
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    emotional_analysis = DeepFace.analyze(img_path, actions=actions, enforce_detection=False)

    if save_emo_frames and (emo_frames_path is not None):
        anno_img = annotate_face_emotion(img_path=img_path,
                              emotional_analysis=emotional_analysis,
                              face_locations=face_locations,
                              text_color=(0,255,0))
        
        fname = os.path.basename(img_path)
        video_dir_name = os.path.split(os.path.normpath(img_path))[0].split("\\")[-1].replace('-frames', '')
        emo_frame_out = os.path.normpath(os.path.join(emo_frames_path, video_dir_name))
        os.makedirs(emo_frame_out, exist_ok=True)
        emo_frame_out = os.path.normpath(os.path.join(emo_frame_out, f"emo_{fname}"))
        cv2.imwrite(emo_frame_out, anno_img)

        if verbose:
            print(f"Saved Emtional Analysis frame to {emo_frame_out}")

    
    return emotional_analysis

def analyze_directory(dir_path: str, 
                      mode: str,
                      face_match_model: str = "Facenet512",
                      face_match_frames: bool = False,
                      face_match_people: bool = False,
                      face_img_path: str = None,
                      save_emo_frames: bool = False,
                      emo_frames_path: str = None,
                      verbose: bool = False) -> pd.DataFrame | None:
    """Analyze emotions in all images in a directory.
    
    Args:
        dir_path: Path to directory containing frame images
        mode: Analysis mode (currently only 'deepface' supported)
        face_match_model: Model to use for face matching
        face_match_frames: Whether to match faces across frames
        face_match_people: Whether to match people across frames
        face_img_path: Path to reference face image(s)
        verbose: Print progress messages
    
    Returns:
        DataFrame with analysis results or None on error
    """
    if not os.path.exists(dir_path):
        print(f"Error: Directory {dir_path} does not exist")
        return None
    
    if emo_frames_path is not None:
        os.makedirs(emo_frames_path, exist_ok=True)

    if mode == 'deepface':
        frame_idx = 0
        frames_df = pd.DataFrame()
        
        if face_match_people:
            faces = {}
            face_idx = 1
            emotional_analysis_out = {
                'frame_id' : [],
                'face_id' : [],
                'dominant_emotion' : [],
                'emotions' : []
                }
        else:
            emotional_analysis_out = {
                'frame_id' : [],
                'dominant_emotion' : [],
                'emotions' : []
                }

        if face_match_frames:
            """If there is a reference face image(s) provided, use it to 
               find all of the frames that have a matching face."""
            if face_img_path is not None:
                for img_path in face_img_path:
                    frames_df.join(find_frames_by_face(face_img_path=face_img_path, frames_path=dir_path))
                frames_df = frames_df.sort_values(by=['identity'])
                imgs_paths = frames_df['identity'].values
            else:
                print("ERROR: No Path to a face image provided. Exiting...")
                return None

        else:
            imgs_paths = crawl_dir(dir=dir_path, ext='.jpg')

        for img_path in tqdm.tqdm(imgs_paths, desc=f"Analyzing {os.path.basename(dir_path)}..."):

            # If using face matching
            if face_match_people:
                face_locations = []
                embeds = create_face_embedding(img_path=img_path, model=face_match_model)
                for embed in embeds:
                    check = check_faces(embed['embedding'], faces, model=face_match_model)
                    if check == None:
                        key = tuple(embed['embedding'])
                        faces[key] = Face(embedding=embed['embedding'],
                                                            face_id=face_idx,
                                                            location=embed['facial_area'])
                        face_locations.append((face_idx, embed['facial_area']))
                        
                        face_idx += 1
                    else:
                        key = tuple(check)
                        faces[key].add_location(embed['facial_area'])
                        face_locations.append((faces[key].id, embed['facial_area']))

                # Perform emotional analysis
                emotional_analysis = deepface_analysis(img_path=img_path, 
                                                       save_emo_frames=save_emo_frames, 
                                                       emo_frames_path=emo_frames_path, 
                                                       face_locations=face_locations,
                                                       verbose=verbose)
                
                nojoy=True
                for face, _ in faces.items():
                    for emo in emotional_analysis:
                        matched = any((emo['region'] == loc and faces[face].id == id) for id, loc in face_locations)
                        if matched:
                            nojoy=False
                            emotional_analysis_out['frame_id'].append(frame_idx)
                            emotional_analysis_out['face_id'].append(faces[face].id)
                            emotional_analysis_out['dominant_emotion'].append(emo['dominant_emotion'])
                            emotional_analysis_out['emotions'].append(emo['emotion'])
                if nojoy and verbose:
                    print(f"\nWarning: No matching faces found in frame {frame_idx}\n")

            else:
                # Perform emotional analysis
                emotional_analysis = deepface_analysis(img_path=img_path, 
                                                       save_emo_frames=save_emo_frames,
                                                       emo_frames_path=emo_frames_path,
                                                       verbose=verbose)
                for emo in emotional_analysis:
                        emotional_analysis_out['frame_id'].append(frame_idx)
                        emotional_analysis_out['dominant_emotion'].append(emo['dominant_emotion'])
                        emotional_analysis_out['emotions'].append(emo['emotion'])
                
            
            frame_idx += 1

        return pd.DataFrame.from_dict(emotional_analysis_out)
    
    else:
        print(f"Error: Unsupported mode '{mode}'")
        return None