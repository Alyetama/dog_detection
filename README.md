# Dog Detection

High-speed dog detection system using YOLOv8 with efficient dataset preparation and batch processing capabilities.

## Overview

This project provides tools for preparing YOLO-format datasets from Label Studio annotations and performing high-speed inference on large image collections using YOLOv8. It's designed for scalable processing of millions of images with SQLite logging and multithreaded I/O operations.

## Features

- **Efficient Dataset Preparation**: Convert Label Studio JSON exports to YOLO format with flexible class handling
- **High-Speed Inference**: Process millions of images with batch processing and GPU acceleration
- **Database Logging**: SQLite integration for tracking predictions and errors
- **Multithreaded I/O**: Concurrent file operations for optimal performance
- **Flexible Sampling**: Support for exhaustive scanning or instant random sampling of large datasets
- **Visualization**: Optional annotation visualization with bounding boxes
- **Hard Negative Mining**: Support for hallucinated/hard negative images in training data

## Installation

### Requirements

- Python 3.7+
- PyTorch with CUDA support (recommended for GPU acceleration)
- Dependencies listed in requirements (inferred from code):
  - `ultralytics` - YOLOv8 implementation
  - `opencv-python` - Image processing
  - `Pillow` - Image handling
  - `tqdm` - Progress bars
  - `boto3` - AWS S3 integration
  - `python-dotenv` - Environment variable management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Alyetama/dog_detection.git
cd dog_detection
```

2. Install dependencies:
```bash
pip install ultralytics opencv-python Pillow tqdm boto3 python-dotenv
```

3. For AWS S3 support, create a `.env` file with your credentials:
```env
ENDPOINT_URL=https://your-s3-endpoint.com
ACCESS_KEY_ID=your_access_key
SECRET_ACCESS_KEY=your_secret_key
BUCKET_NAME=your_bucket_name
BUCKET_REGION=us-east-1
```

## Usage

### 1. Dataset Preparation

Prepare a YOLO-format dataset from Label Studio exports:

```bash
python prepare_detection_yolo_dataset.py \
  -f exported_project.json \
  -l annotations_group_name \
  -i local_images_dir \
  --background \
  --hallucinations hard_negatives_dir
```

**Arguments:**
- `-f, --project-exported-file`: Path to Label Studio JSON export (required)
- `-l, --label-by`: Parent group name containing labels (required)
- `-i, --images-dir`: Local directory to copy images from (optional, avoids S3 downloads)
- `-e, --exclude-classes`: Comma-separated list of classes to exclude
- `--single-class`: Treat all classes as single target class (id 0)
- `--background`: Include background images (up to 10% of dataset)
- `--hallucinations`: Path to directory with hard negative images

**Output:**
- `project_folder/`: Dataset directory with train/val splits
- `dataset.yaml`: YOLO format configuration file
- `project_folder/classes.txt`: Class mapping

### 2. High-Speed Inference

Run predictions on large image collections:

```bash
python predict_and_filter.py \
  -w best.pt \
  -s /path/to/images \
  -o output_dir \
  --conf 0.25 \
  --imgsz 1280 \
  --batch 8 \
  --db predictions.db \
  --copy \
  --visualize
```

**Arguments:**
- `-w, --weights`: Path to YOLO weights file (required)
- `-s, --source`: Base directory to scan (required)
- `-o, --output`: Output directory for detections (required)
- `-p, --pattern`: Optional glob pattern for subdirectories
- `-c, --conf`: Confidence threshold (default: 0.25)
- `--imgsz`: Inference image size (default: 1280)
- `--batch`: Batch size for GPU inference (default: 8)
- `--half`: Enable FP16 half-precision inference
- `--db`: SQLite database path for logging predictions and errors
- `--copy`: Copy detected images to output directory
- `--visualize`: Save annotated images with bounding boxes
- `--select-random N`: Randomly select N unprocessed images (instant sampling for millions)
- `--reindex`: Force rebuild of image index file

**Output:**
- `output_dir/`: Directory containing detected images
- `output_dir_visualized/`: Annotated images (if `--visualize` enabled)
- `predictions.db`: SQLite database with:
  - `predictions` table: Image path, region info, detection count, confidence scores
  - `errors` table: Failed/unreadable images

### Dataset Index

The first random selection run builds an instant index file (`dataset_image_index.txt`) that enables millisecond-level random sampling of millions of files in subsequent runs.

## Project Structure

```
dog_detection/
├── prepare_detection_yolo_dataset.py   # Dataset preparation script
├── predict_and_filter.py               # High-speed inference script
├── LICENSE                             # MIT License
└── README.md
```

## Performance

- **Inference Speed**: Processes thousands of images per minute on modern GPUs
- **Memory Efficiency**: Streaming inference with configurable batch sizes
- **Random Sampling**: Instant sampling of millions of files via indexed lookup
- **Scalability**: Tested on datasets with millions of images

## Example Workflows

### Prepare dataset from Label Studio with S3

```bash
python prepare_detection_yolo_dataset.py \
  -f annotations.json \
  -l annotations \
  --background \
  --single-class
```

### Scan directory for dogs and copy detections

```bash
python predict_and_filter.py \
  -w weights/best.pt \
  -s /mnt/data/images \
  -o results/detected_dogs \
  --copy \
  --db results/detections.db \
  --batch 16 \
  --half
```

### Randomly sample and process 10,000 images

```bash
python predict_and_filter.py \
  -w weights/best.pt \
  -s /mnt/large_dataset \
  -o results/sampled \
  --select-random 10000 \
  --db results/predictions.db \
  --visualize
```

## Database Schema

### predictions table
```sql
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  region_name TEXT,
  parent_region_name TEXT,
  image_full_path TEXT UNIQUE,
  image_id TEXT,
  num_detections INTEGER,
  conf_per_detection TEXT
);
```

### errors table
```sql
CREATE TABLE errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_full_path TEXT UNIQUE,
  image_id TEXT,
  region_name TEXT,
  parent_region_name TEXT,
  error_message TEXT
);
```

## Requirements

- **Primary Language**: Python
- **License**: MIT
- **Dependencies**: ultralytics, opencv-python, Pillow, tqdm, boto3, python-dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve this project.
