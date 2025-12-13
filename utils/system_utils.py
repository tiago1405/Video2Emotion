import os
import tqdm
from utils.video_utils import video_to_images


def crawl_dir(dir: str, ext: str) -> list[str]:
    """Find all files with given extension in a directory.
    
    Args:
        dir: Directory path to search
        ext: File extension to match (with or without dot)
    
    Returns:
        List of full paths to matching files
    """
    files = []
    if not ext.startswith('.'):
        ext = f'.{ext}'
    
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path) and file.endswith(ext):
            files.append(file_path)
    return files

def process_dir(videos_dir: str, 
                every_n: float, 
                output_dir: str = "./", 
                time_or_frames: str = "frames", 
                videos_ext: str = "mp4", 
                verbose: bool = False) -> None:
    """Process all videos in a directory and convert them to 
    frames.
    
    Args:
        videos_dir: Path to directory containing videos
        every_n: Save frames every n frames or seconds
        output_dir: Directory to save extracted frames
        time_or_frames: Extract by 'time' (in seconds) or 'frames'
        videos_ext: Video file extension to process
        verbose: Print progress messages
    """
    if not os.path.exists(videos_dir):
        print(f"Error: Directory {videos_dir} does not exist")
        return
    
    video_paths = crawl_dir(dir=videos_dir, ext=videos_ext)
    
    if not video_paths:
        print(f"No videos with extension .{videos_ext} found in {videos_dir}")
        return
    
    for video_path in tqdm.tqdm(video_paths, desc=f"Converting videos to frames..."):
        video_to_images(video_path=video_path, out_dir=output_dir, every_n=every_n, time_or_frames=time_or_frames)
    
    if verbose:
        print(f"Done converting videos in the {videos_dir} directory!")