"""
Video to Emotion Analysis Pipeline.

This script processes videos in a directory, extracts frames at set intervals,
and analyzes facial emotions using DeepFace. It supports face matching and
outputs results as either multiple CSVs or one large CSV.
"""

import os
import sys
import getopt
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

from utils.system_utils import process_dir
from video2emo import analyze_directory
# Add to main.py after the imports
from utils.statistics_utils import (
    generate_summary_report,
    generate_statistics_csv,
    print_statistics_summary
)


def main(input_dir: str,
         frames_dir: str,
         output_dir: str, 
         every_n: float,
         time_or_frames: str, 
         file_ext: str,
         mode: str = "deepface",
         face_match_model: str = "Facenet512",
         face_match_frames: bool = False,
         face_match_people: bool = False,
         face_img_path: str = None,
         multi_csv: bool = True,
         index: bool = False,
         save_emo_frames: bool = False,
         emo_frames_path: str = None,
         generate_stats: bool = False,
         stats_format: str = "csv",
         verbose: bool = False) -> None:
    """Process videos into frames and analyzes facial emotions in frames.
    
    Args:
        input_dir: Directory containing input videos
        frames_dir: Directory to store extracted frames
        output_dir: Directory for output CSV files
        every_n: Extract every n frames or seconds
        time_or_frames: Extract by 'time' or 'frames'
        file_ext: Video file extension to process
        mode: Analysis mode (default: 'deepface')
        face_match_model: Model for face matching
        face_match_frames: Enable face matching across frames
        face_match_people: Enable face matching for people identification
        face_img_path: Path to reference face image
        multi_csv: Output separate CSV for each video (True) or combined (False)
        index: Include index column in CSV output
        save_emo_frames: Save emotion annotated frames
        emo_frames_path: Directory to save emotion annotated frames
        generate_stats: Generate statistics reports
        stats_format: Statistics format ('csv', 'json', or 'console')
        verbose: Print detailed progress messages
    """
    video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # Create directories if they don't exist
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all dataframes for statistics
    df_out_list = []

    # if len(video_dirs) > 1:
    #     for video_dir in video_dirs:
    #         video_dir_path = os.path.join(input_dir, video_dir)
    #         process_dir(videos_dir=video_dir_path, 
    #                     every_n=every_n,
    #                     time_or_frames=time_or_frames, 
    #                     output_dir=frames_dir,
    #                     videos_ext=file_ext)
    #         if verbose:
    #             print(f"Completed processing of {video_dir}")
    # elif video_dirs == []:
    #     process_dir(videos_dir=input_dir, 
    #                     every_n=every_n,
    #                     time_or_frames=time_or_frames, 
    #                     output_dir=frames_dir,
    #                     videos_ext=file_ext)
    #     if verbose:
    #         print(f"Completed processing of {input_dir}")
    # else:
    #     print(f"Couldn't find any input directory(ies) at {input_dir}. Please verify and try again.")
    #     return None
    
    dirs_to_analyze = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]

    for frame_dir in dirs_to_analyze:
        frame_dir_path = os.path.join(frames_dir, frame_dir)
        df_out = analyze_directory(dir_path=frame_dir_path,
                        mode=mode,
                        face_match_model=face_match_model,
                        face_match_frames=face_match_frames,
                        face_match_people=face_match_people,
                        face_img_path=face_img_path,
                        save_emo_frames=save_emo_frames,
                        emo_frames_path=emo_frames_path,
                        verbose=verbose)
        if verbose:
            print(f"Done analyzing {frame_dir}.")
        
        if df_out is not None:
            # Add source directory to keep track of video
            df_out['source_directory'] = os.path.basename(frame_dir)
            df_out_list.append(df_out)
            
            
            # Save individual CSV per-video
            if multi_csv:
                video_name = os.path.basename(frame_dir).replace('-frames', '')
                output_dir_path = os.path.join(output_dir, video_name)
                os.makedirs(output_dir_path, exist_ok=True)
                output_file_path = os.path.join(output_dir_path, f"{os.path.basename(frame_dir)}.csv")
                df_out.to_csv(output_file_path, index=index)
                
                # Generate per-video statistics
                if generate_stats:
                    if stats_format == "csv":
                        stats_files = generate_statistics_csv(
                            df_out, output_dir_path, prefix=f"{video_name}_stats"
                        )
                        if verbose:
                            print(f"Generated {len(stats_files)} statistics files for {video_name}")
                    elif stats_format == "json":
                        stats_path = os.path.join(output_dir_path, f"{video_name}_statistics.json")
                        generate_summary_report(df_out, output_path=stats_path)
                        if verbose:
                            print(f"Statistics report saved to: {stats_path}")

    # Generate combined statistics regardless of multi_csv setting
    if df_out_list and generate_stats:
        df_combined = pd.concat(df_out_list, ignore_index=True)
        
        if not multi_csv:
            # Save combined CSV if not using multi_csv
            output_path = os.path.join(output_dir, "combined_output.csv")
            df_combined.to_csv(output_path, index=index)
        
        # Generate combined statistics
        if stats_format == "console":
            print("\n" + "="*60)
            print("COMBINED STATISTICS (ALL VIDEOS)")
            print("="*60)
            print_statistics_summary(df_combined)
        elif stats_format == "json":
            stats_path = os.path.join(output_dir, "combined_statistics.json")
            generate_summary_report(df_combined, output_path=stats_path)
            if verbose:
                print(f"Combined statistics report saved to: {stats_path}")
        else:  # csv
            stats_files = generate_statistics_csv(df_combined, output_dir, prefix="combined")
            if verbose:
                print(f"\nGenerated {len(stats_files)} combined statistics files:")
                for f in stats_files:
                    print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    options = "hv"
    long_options = ["fm_frames", "fm_people", "verbose", "multi_csv", "index", "save_emo_frames", "emo_frames_path=",
                    "fm_model=", "mode=", "face_img_path=", "frames_dir=", "output_dir=", 
                    "input_dir=", "every_n=", "time_or_frames=", "file_ext=",
                    "generate_stats", "stats_format="]
    
    # Default values
    input_dir = "./videos"
    frames_dir = "./frames"
    output_dir = "./"
    every_n = 1.0
    time_or_frames = "frames"
    file_ext = ".mp4"
    mode = "deepface"
    face_match_model = "Facenet512"
    face_match_frames = False
    face_match_people = False
    face_img_path = None
    multi_csv = True
    index = False
    save_emo_frames = False
    emo_frames_path = None
    generate_stats = False
    stats_format = "csv"
    verbose = False
    
    try:
        opts, args = getopt.getopt(args, options, long_options)
    except getopt.GetoptError:
        print("Error parsing arguments")
        sys.exit(2)
    
    for opt, arg in opts:
        match opt:
            case "-h":
                print("Usage: script.py [options]")
                print("Options:")
                print("  -h                    Show this help message")
                print("  -v                    Verbose mode")
                print("  --input_dir=          Input directory containing videos (default: ./videos)")
                print("  --frames_dir=         Output directory for frames (default: ./frames)")
                print("  --output_dir=         Output directory for results (default: ./)")
                print("  --every_n=            Extract every n frames/seconds (default: 1.0)")
                print("  --time_or_frames=     Extract by 'time' or 'frames' (default: frames)")
                print("  --file_ext=           Video file extension (default: .mp4)")
                print("  --mode=               Analysis mode (default: deepface)")
                print("  --fm_model=           Face matching model (default: Facenet512)")
                print("  --fm_frames           Enable face matching across frames")
                print("  --fm_people           Enable face matching across people in frames")
                print("  --face_img_path=      Path to reference face image")
                print("  --multi_csv           Output multiple CSV files (one per video, default: True)")
                print("  --index               Include index in CSV output (default: False)")
                print("  --save_emo_frames     Save emotion annotated frames (default: False)")
                print("  --emo_frames_path=    Directory to save emotion annotated frames")
                print("  --generate_stats      Generate statistics reports (default: False)")
                print("  --stats_format=       Statistics format: csv, json, or console (default: csv)")
                sys.exit(0)
            case "-v":
                verbose = True
            case "--input_dir":
                input_dir = arg
            case "--frames_dir":
                frames_dir = arg
            case "--output_dir":
                output_dir = arg
            case "--every_n":
                every_n = float(arg)
            case "--time_or_frames":
                time_or_frames = arg
            case "--file_ext":
                file_ext = arg
            case "--mode":
                mode = arg
            case "--fm_model":
                face_match_model = arg
            case "--fm_frames":
                face_match_frames = True
            case "--fm_people":
                face_match_people = True
            case "--face_img_path":
                face_img_path = arg
            case "--verbose":
                verbose = True
            case "--multi_csv":
                multi_csv = True
            case "--index":
                index = True
            case "--save_emo_frames":
                save_emo_frames = True
            case "--emo_frames_path":
                emo_frames_path = arg
            case "--generate_stats":
                generate_stats = True
            case "--stats_format":
                stats_format = arg
    
    # Call main function with parsed arguments
    main(input_dir=input_dir,
         frames_dir=frames_dir,
         output_dir=output_dir,
         every_n=every_n,
         time_or_frames=time_or_frames,
         file_ext=file_ext,
         mode=mode,
         face_match_model=face_match_model,
         face_match_frames=face_match_frames,
         face_match_people=face_match_people,
         face_img_path=face_img_path,
         multi_csv=multi_csv,
         index=index,
         save_emo_frames=save_emo_frames,
         emo_frames_path=emo_frames_path,
         generate_stats=generate_stats,
         stats_format=stats_format,
         verbose=verbose)