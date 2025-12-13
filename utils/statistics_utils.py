"""
Statistics generation for emotion analysis results.

This module provides functions to generate useful statistics from emotion
analysis data, including emotion distributions, temporal patterns, and
comparative analysis across videos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os


def calculate_emotion_distribution(df: pd.DataFrame, 
                                   group_by: str = None) -> pd.DataFrame:
    """Calculate the distribution of emotions across all frames.
    
    Args:
        df: DataFrame with emotion analysis results
        group_by: Optional column to group by (e.g., 'source_directory', 'face_id')
    
    Returns:
        DataFrame with emotion counts and percentages
    """
    if df.empty:
        return pd.DataFrame()
    
    if group_by and group_by in df.columns:
        # Group by specified column
        emotion_counts = df.groupby([group_by, 'dominant_emotion']).size().reset_index(name='count')
        
        # Calculate percentages within each group
        totals = emotion_counts.groupby(group_by)['count'].sum()
        emotion_counts['percentage'] = emotion_counts.apply(
            lambda row: (row['count'] / totals[row[group_by]]) * 100, 
            axis=1
        )
    else:
        # Overall distribution
        emotion_counts = df['dominant_emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        emotion_counts['percentage'] = (emotion_counts['count'] / len(df)) * 100
    
    return emotion_counts.sort_values('count', ascending=False)


def calculate_temporal_patterns(df: pd.DataFrame, 
                               window_size: int = 10,
                               group_by: str = None) -> pd.DataFrame:
    """Analyze temporal patterns of emotions over time.
    
    Args:
        df: DataFrame with emotion analysis results
        window_size: Number of frames for rolling window analysis
        group_by: Optional column to group by (e.g., 'source_directory')
    
    Returns:
        DataFrame with temporal emotion patterns
    """
    if df.empty or 'frame_id' not in df.columns:
        return pd.DataFrame()
    
    # Sort by frame_id
    df_sorted = df.sort_values('frame_id').copy()
    
    if group_by and group_by in df.columns:
        results = []
        for group_name, group_df in df_sorted.groupby(group_by):
            temporal_stats = _calculate_group_temporal_stats(
                group_df, window_size, group_name
            )
            results.append(temporal_stats)
        
        return pd.concat(results, ignore_index=True)
    else:
        return _calculate_group_temporal_stats(df_sorted, window_size)


def _calculate_group_temporal_stats(df: pd.DataFrame, 
                                    window_size: int,
                                    group_name: str = None) -> pd.DataFrame:
    """Calculate temporal statistics for a single group."""
    # Create emotion change indicator
    df = df.copy()
    df['emotion_changed'] = (df['dominant_emotion'] != df['dominant_emotion'].shift(1)).astype(int)

    if not df.empty:
        df.iloc[0, df.columns.get_loc('emotion_changed')] = 0
    
    # Calculate rolling statistics
    df['emotion_stability'] = df['emotion_changed'].rolling(
        window=window_size, min_periods=1
    ).apply(lambda x: 1 - x.mean())
    
    # Calculate emotion duration (consecutive frames with same emotion)
    df['emotion_duration'] = df.groupby(
        (df['dominant_emotion'] != df['dominant_emotion'].shift()).cumsum()
    ).cumcount() + 1
    
    temporal_stats = pd.DataFrame({
        'frame_id': df['frame_id'],
        'dominant_emotion': df['dominant_emotion'],
        'emotion_stability': df['emotion_stability'],
        'emotion_duration': df['emotion_duration'],
        'emotion_changed': df['emotion_changed']
    })
    
    if group_name:
        temporal_stats['group'] = group_name
    
    return temporal_stats


def calculate_emotion_transitions(df: pd.DataFrame,
                                 group_by: str = None) -> pd.DataFrame:
    """Calculate emotion transition matrix.
    
    Args:
        df: DataFrame with emotion analysis results
        group_by: Optional column to group by
    
    Returns:
        DataFrame showing transition probabilities between emotions
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    
    df_sorted = df.sort_values('frame_id').copy()
    
    if group_by and group_by in df.columns:
        results = []
        for group_name, group_df in df_sorted.groupby(group_by):
            transitions = _calculate_group_transitions(group_df, group_name)
            results.append(transitions)
        
        return pd.concat(results, ignore_index=True)
    else:
        return _calculate_group_transitions(df_sorted)


def _calculate_group_transitions(df: pd.DataFrame, 
                                group_name: str = None) -> pd.DataFrame:
    """Calculate transition matrix for a single group."""
    df = df.copy()
    df['next_emotion'] = df['dominant_emotion'].shift(-1)
    
    # Remove last row (no next emotion)
    df = df[:-1]
    
    # Count transitions
    transitions = df.groupby(['dominant_emotion', 'next_emotion']).size().reset_index(name='count')
    
    # Calculate probabilities
    totals = transitions.groupby('dominant_emotion')['count'].sum()
    transitions['probability'] = transitions.apply(
        lambda row: row['count'] / totals[row['dominant_emotion']], 
        axis=1
    )
    
    if group_name:
        transitions['group'] = group_name
    
    return transitions


def compare_videos(df: pd.DataFrame, 
                  video_column: str = 'source_directory') -> pd.DataFrame:
    """Compare emotion distributions across different videos.
    
    Args:
        df: DataFrame with emotion analysis results
        video_column: Column containing video identifiers
    
    Returns:
        DataFrame with comparative statistics
    """
    if df.empty or video_column not in df.columns:
        return pd.DataFrame()
    
    comparisons = []
    
    for video in df[video_column].unique():
        video_df = df[df[video_column] == video]
        
        stats = {
            'video': video,
            'total_frames': len(video_df),
            'unique_emotions': video_df['dominant_emotion'].nunique(),
            'most_common_emotion': video_df['dominant_emotion'].mode()[0] if len(video_df) > 0 else None,
            'emotion_changes': (video_df['dominant_emotion'] != video_df['dominant_emotion'].shift()).sum() - 1,
        }
        
        # Add percentage for each emotion
        emotion_dist = video_df['dominant_emotion'].value_counts(normalize=True) * 100
        for emotion, pct in emotion_dist.items():
            stats[f'{emotion}_pct'] = round(pct, 2)
        
        comparisons.append(stats)
    
    return pd.DataFrame(comparisons)


def calculate_face_statistics(df: pd.DataFrame,
                             face_column: str = 'face_id') -> pd.DataFrame:
    """Calculate statistics per face/person.
    
    Args:
        df: DataFrame with emotion analysis results
        face_column: Column containing face identifiers
    
    Returns:
        DataFrame with per-face statistics
    """
    if df.empty or face_column not in df.columns:
        return pd.DataFrame()
    
    face_stats = []
    
    for face_id in df[face_column].unique():
        face_df = df[df[face_column] == face_id]
        
        stats = {
            'face_id': face_id,
            'total_appearances': len(face_df),
            'dominant_emotion': face_df['dominant_emotion'].mode()[0] if len(face_df) > 0 else None,
            'emotion_variety': face_df['dominant_emotion'].nunique(),
            'average_emotion_duration': _calculate_avg_emotion_duration(face_df),
        }
        
        # Add emotion breakdown
        emotion_dist = face_df['dominant_emotion'].value_counts()
        for emotion, count in emotion_dist.items():
            stats[f'{emotion}_count'] = count
            stats[f'{emotion}_pct'] = round((count / len(face_df)) * 100, 2)
        
        face_stats.append(stats)
    
    return pd.DataFrame(face_stats)


def _calculate_avg_emotion_duration(df: pd.DataFrame) -> float:
    """Calculate average duration of each emotion."""
    if df.empty:
        return 0.0
    
    df = df.sort_values('frame_id').copy()
    df['emotion_group'] = (df['dominant_emotion'] != df['dominant_emotion'].shift()).cumsum()
    durations = df.groupby('emotion_group').size()
    
    return durations.mean()


def generate_summary_report(df: pd.DataFrame,
                           output_path: str = None) -> Dict:
    """Generate comprehensive summary report.
    
    Args:
        df: DataFrame with emotion analysis results
        output_path: Optional path to save report as JSON
    
    Returns:
        Dictionary containing summary statistics
    """
    if df.empty:
        return {'error': 'Empty DataFrame'}
    
    report = {
        'overview': {
            'total_frames': len(df),
            'unique_emotions': df['dominant_emotion'].nunique(),
            'emotions_detected': df['dominant_emotion'].unique().tolist(),
        },
        'emotion_distribution': calculate_emotion_distribution(df).to_dict('records'),
    }
    
    # Add video comparison if source_directory exists
    if 'source_directory' in df.columns:
        report['video_comparison'] = compare_videos(df).to_dict('records')
        report['emotion_by_video'] = calculate_emotion_distribution(
            df, group_by='source_directory'
        ).to_dict('records')
    
    # Add face statistics if face_id exists
    if 'face_id' in df.columns:
        report['face_statistics'] = calculate_face_statistics(df).to_dict('records')
    
    # Add temporal analysis
    temporal_data = calculate_temporal_patterns(df)
    if not temporal_data.empty:
        report['temporal_patterns'] = {
            'average_stability': temporal_data['emotion_stability'].mean(),
            'average_emotion_duration': temporal_data['emotion_duration'].mean(),
            'total_emotion_changes': temporal_data['emotion_changed'].sum(),
        }
    
    # Add transition matrix
    transitions = calculate_emotion_transitions(df)
    if not transitions.empty:
        report['emotion_transitions'] = transitions.to_dict('records')
    
    # Save to file if path provided
    if output_path:
        import json

        def _to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=_to_serializable)
    
    return report


def generate_statistics_csv(df: pd.DataFrame, 
                            output_dir: str,
                            prefix: str = "stats") -> List[str]:
    """Generate multiple CSV files with different statistics.
    
    Args:
        df: DataFrame with emotion analysis results
        output_dir: Directory to save CSV files
        prefix: Prefix for output filenames
    
    Returns:
        List of paths to generated CSV files
    """
    if df.empty:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 1. Emotion distribution
    emotion_dist = calculate_emotion_distribution(df)
    if not emotion_dist.empty:
        path = os.path.join(output_dir, f"{prefix}_emotion_distribution.csv")
        emotion_dist.to_csv(path, index=False)
        generated_files.append(path)
    
    # 2. Temporal patterns
    temporal = calculate_temporal_patterns(df)
    if not temporal.empty:
        path = os.path.join(output_dir, f"{prefix}_temporal_patterns.csv")
        temporal.to_csv(path, index=False)
        generated_files.append(path)
    
    # 3. Emotion transitions
    transitions = calculate_emotion_transitions(df)
    if not transitions.empty:
        path = os.path.join(output_dir, f"{prefix}_emotion_transitions.csv")
        transitions.to_csv(path, index=False)
        generated_files.append(path)
    
    # 4. Video comparison (if applicable)
    if 'source_directory' in df.columns:
        video_comp = compare_videos(df)
        if not video_comp.empty:
            path = os.path.join(output_dir, f"{prefix}_video_comparison.csv")
            video_comp.to_csv(path, index=False)
            generated_files.append(path)
    
    # 5. Face statistics (if applicable)
    if 'face_id' in df.columns:
        face_stats = calculate_face_statistics(df)
        if not face_stats.empty:
            path = os.path.join(output_dir, f"{prefix}_face_statistics.csv")
            face_stats.to_csv(path, index=False)
            generated_files.append(path)
    
    return generated_files


def print_statistics_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of statistics to console.
    
    Args:
        df: DataFrame with emotion analysis results
    """
    if df.empty:
        print("No data to analyze.")
        return
    
    print("\n" + "="*60)
    print("EMOTION ANALYSIS STATISTICS SUMMARY")
    print("="*60 + "\n")
    
    # Overview
    print(f"Total Frames Analyzed: {len(df)}")
    print(f"Unique Emotions Detected: {df['dominant_emotion'].nunique()}")
    print(f"Emotions: {', '.join(df['dominant_emotion'].unique())}\n")
    
    # Emotion distribution
    print("EMOTION DISTRIBUTION:")
    print("-" * 40)
    emotion_dist = df['dominant_emotion'].value_counts()
    for emotion, count in emotion_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {emotion:15s}: {count:5d} frames ({pct:5.1f}%)")
    
    # Video comparison if available
    if 'source_directory' in df.columns:
        print("\n\nVIDEO COMPARISON:")
        print("-" * 40)
        video_comp = compare_videos(df)
        for _, row in video_comp.iterrows():
            print(f"\n  Video: {row['video']}")
            print(f"    Frames: {row['total_frames']}")
            print(f"    Most Common: {row['most_common_emotion']}")
            print(f"    Emotion Changes: {row['emotion_changes']}")
    
    # Face statistics if available
    if 'face_id' in df.columns:
        print("\n\nFACE/PERSON STATISTICS:")
        print("-" * 40)
        face_stats = calculate_face_statistics(df)
        for _, row in face_stats.iterrows():
            print(f"\n  Face ID: {row['face_id']}")
            print(f"    Appearances: {row['total_appearances']}")
            print(f"    Dominant Emotion: {row['dominant_emotion']}")
            print(f"    Emotion Variety: {row['emotion_variety']}")
    
    # Temporal patterns
    temporal = calculate_temporal_patterns(df)
    if not temporal.empty:
        print("\n\nTEMPORAL PATTERNS:")
        print("-" * 40)
        print(f"  Average Emotion Stability: {temporal['emotion_stability'].mean():.2f}")
        print(f"  Average Emotion Duration: {temporal['emotion_duration'].mean():.2f} frames")
        print(f"  Total Emotion Changes: {temporal['emotion_changed'].sum()}")
    
    print("\n" + "="*60 + "\n")