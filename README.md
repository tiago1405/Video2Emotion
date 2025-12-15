# Video to Emotion Analysis Pipeline

A Python-based pipeline for extracting frames from videos and analyzing facial emotions using DeepFace. Supports face matching/tracking, emotion analysis, and optional statistics reporting.

## Features
- Video frame extraction at fixed frame or time intervals
- Emotion analysis via DeepFace
- Face matching / tracking across frames
- Optional statistics (CSV/JSON/console)
- Batch processing of many videos in one run (sequential)
- Docker workflow (if you build the images)

## Installation

### Prerequisites

- Python 3.12+
- Tested library versions:
  - deepface 0.0.96
  - tensorflow 2.20.0, tf-keras 2.20.1
  - opencv-python 4.12.0.88
  - pandas 2.3.3, numpy 2.2.6, tqdm 4.67.1
  - mtcnn 1.0.0, retina-face 0.0.17
  - pillow 12.0.0, protobuf 6.33.2, h5py 3.15.1

### Clone repository
```bash
git clone https://github.com/tiago1405/Video2Emotion.git
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Docker (optional)
```bash
docker-compose build
```

## Directory Structure
```
project/
├── videos/              # Input videos
├── frames/              # Extracted frames
├── output/              # CSV/stat outputs
├── emotion_frames/      # Annotated frames (optional)
├── main.py
├── video2emo.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── utils/
  ├── face_matching.py
  ├── statistics_utils.py
  ├── system_utils.py
  └── video_utils.py
```

## Usage

### Command Line Usage

Process videos with default settings:
```bash
python main.py --input_dir=./videos --frames_dir=./frames --output_dir=./output
```

### Extract Frames Every N Frames
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --every_n=5.0 \
  --time_or_frames=frames
```

### Extract Frames Every N Seconds
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --every_n=1.0 \
  --time_or_frames=time
```

### Enable Face Tracking Across Frames

Track individual faces across all frames, assigning a face id to them:
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --fm_people \
  --fm_model=Facenet512
```
Useful to distinguish which person displayed what emotion.

### Match Specific Face in Videos

Find all frames containing a specific person's face:
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --fm_frames \
  --face_img_path=./reference_face.jpg
```
Useful when conducting research about specific people's emotional responses.

### Save Annotated Emotion Frames

Save frames with emotion annotations and a bounding box overlaid on faces in frames:
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --save_emo_frames \
  --emo_frames_path=./emotion_frames
```

### Generate Statistics Reports

Generate comprehensive statistics in different formats:

#### CSV Format (Multiple Files)
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --generate_stats \
  --stats_format=csv
```

#### JSON Format (Single Report File)
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --generate_stats \
  --stats_format=json
```

#### Console Output
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --generate_stats \
  --stats_format=console
```

### Combined Example: Full Analysis with All Features
```bash
python main.py \
  --input_dir=./videos \
  --frames_dir=./frames \
  --output_dir=./output \
  --every_n=1.0 \
  --time_or_frames=frames \
  --file_ext=.mp4 \
  --fm_people \
  --fm_model=Facenet512 \
  --save_emo_frames \
  --emo_frames_path=./emotion_frames \
  --generate_stats \
  --stats_format=csv \
  --multi_csv \
  --verbose
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h` | Show help message | - |
| `-v, --verbose` | Enable verbose output | False |
| `--input_dir` | Input directory containing videos | ./videos |
| `--frames_dir` | Output directory for extracted frames | ./frames |
| `--output_dir` | Output directory for CSV results | ./ |
| `--every_n` | Extract every n frames/seconds | 1.0 |
| `--time_or_frames` | Extract by 'time' or 'frames' | frames |
| `--file_ext` | Video file extension to process | .mp4 |
| `--mode` | Analysis mode | deepface |
| `--fm_model` | Face matching model | Facenet512 |
| `--fm_frames` | Enable finding frames by matching the faces in the frame to (a) reference image(s) | False |
| `--fm_people` | Enable tracking the same face(s) across frames using a face_id value | False |
| `--face_img_path` | Path to reference face image | None |
| `--multi_csv` | Output separate CSV per video | True |
| `--index` | Include index in CSV output | False |
| `--save_emo_frames` | Save emotion-annotated frames | False |
| `--emo_frames_path` | Directory for annotated frames | None |
| `--generate_stats` | Generate statistics reports | False |
| `--stats_format` | Statistics format (csv/json/console) | csv |

## Docker Usage

### Build and Run
```bash
docker-compose up
```

### Connecting to the container
To connect to the container and run code from within the container I reccomend you execute:
```bash
# Start container in the background
docker-compose up -d
# Find the CONTAINER_ID 
docker ps
# Connect to the bash shell using the CONTAINER_ID 
docker exec -it <<CONTAINER_ID>> /bin/bash
```

### Custom Docker Command
You can also add a command to `docker-compose.yml` to automatically run the pipeline with your variables when using `docker-compose up`. For example:
```yaml
command: > 
  python main.py
  --input_dir=/app/videos
  --frames_dir=/app/frames
  --output_dir=/app/output
  --every_n=2.0
  --generate_stats
  --stats_format=json
```

## Output Format

### Output CSV Structure

The main output CSV contains:
- `frame_id`: Frame number
- `dominant_emotion`: Most prominent emotion detected
- `emotions`: Dictionary of all emotion scores
- `source_directory`: Source video name
- `face_id`: Face identifier (if face tracking enabled)

### Statistics Output

When `--generate_stats` is enabled, you get:

#### CSV Format
- `*_emotion_distribution.csv`: Emotion counts and percentages
- `*_temporal_patterns.csv`: Emotion stability and duration over time
- `*_emotion_transitions.csv`: Transition probabilities between emotions
- `*_video_comparison.csv`: Comparative statistics across videos
- `*_face_statistics.csv`: Per-face emotion statistics (if face tracking enabled)

#### JSON Format
A comprehensive report containing:
- Overview statistics
- Emotion distribution
- Video comparisons
- Face statistics
- Temporal patterns
- Emotion transitions

## Supported Face Matching Models

- Facenet512 (default, recommended)
- Facenet
- VGG-Face
- OpenFace
- DeepFace
- DeepID
- ArcFace
- Dlib
- SFace

## Tests
Unit tests are in the `/tests/` directory. These can be run using:
```bash
# With pytest (recommended)
pip install pytest pytest-cov
pytest tests/ -v

# Or via unittest
python -m unittest discover -s tests -p "test_*.py" -v
```
Tests also run automatically through Git workflows as part of CI/CD on code commits.

<!--
## Troubleshooting

### No faces detected
- Ensure video quality is sufficient
- Try adjusting frame extraction frequency
- Check that faces are visible and well-lit

### Memory issues
- Process videos in smaller batches
- Increase frame extraction interval (`--every_n`)
- Use Docker with memory limits

### Slow processing
- Reduce sampling frequency
- Disable face tracking if not needed
- GPU/TensorRT helps when configured; CPU-only is supported

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
 -->
## References

- S. Serengil and A. Ozpinar, "A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules," *Journal of Information Technologies*, vol. 17, no. 2, pp. 95-107, 2024.
- S. I. Serengil and A. Ozpinar, "LightFace: A Hybrid Deep Face Recognition Framework," *2020 Innovations in Intelligent Systems and Applications Conference (ASYU)*, 2020, pp. 23-27.
- S. I. Serengil and A. Ozpinar, "HyperExtended LightFace: A Facial Attribute Analysis Framework," *2021 International Conference on Engineering and Emerging Technologies (ICEET)*, 2021, pp. 1-4.