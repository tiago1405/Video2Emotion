import os
import cv2

def video_to_images(video_path: str, out_dir: str, every_n: float = 24.0, time_or_frames: str = "frames") -> None:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to input video file
        out_dir: Directory to save extracted frames
        every_n: Extract every n frames or seconds
        time_or_frames: Extract by 'time' or 'frames'
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_fps = video.get(cv2.CAP_PROP_FPS)
        
        if video_fps == 0:
            print(f"Error: Invalid FPS for video {video_path}")
            return
        
        frame_idx = 0
        video_out_dir = os.path.join(out_dir, f"{video_name}-frames")
        os.makedirs(video_out_dir, exist_ok=True)

        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame_idx += 1

            if time_or_frames == "frames" and frame_idx % int(every_n) == 0:
                out_file = os.path.join(video_out_dir, f"frame_{int(frame_idx/every_n):05d}.jpg")
                cv2.imwrite(out_file, frame)
            elif time_or_frames == "time" and frame_idx % int(video_fps * every_n) == 0:
                out_file = os.path.join(video_out_dir, f"frame_{int(frame_idx / (video_fps * every_n)):05d}.jpg")
                cv2.imwrite(out_file, frame)
    finally:
        video.release()
