# Dog Detection

High-speed dog detection system using YOLOv26 with efficient dataset preparation, batch inference, and Label Studio integration.

## ✨ Features

- **Dataset Preparation** – Convert Label Studio annotations to YOLO format with flexible class handling
- **High-Speed Inference** – Process millions of images with GPU acceleration and instant random sampling
- **Label Studio Integration** – REST API for real-time predictions and automated task generation
- **Database Logging** – SQLite tracking of predictions and errors
- **Image Optimization** – Optional compression and resizing during dataset prep
- **Hard Negative Mining** – Support for hallucinated/false-positive images in training

## 🚀 Quick Start

<details open>
<summary><b>Installation</b></summary>

```bash
git clone https://github.com/Alyetama/dog_detection.git
cd dog_detection
pip install -r requirements.txt
```

For Label Studio integration, also set up `.env`:
```env
LABEL_STUDIO_TOKEN=your_token
```

For S3 support, add to `.env`:
```env
ENDPOINT_URL=https://your-s3-endpoint.com
ACCESS_KEY_ID=your_key
SECRET_ACCESS_KEY=your_secret
BUCKET_NAME=your_bucket
BUCKET_REGION=us-east-1
```

</details>

## 📋 Scripts

### 1️⃣ Prepare Dataset

Convert Label Studio JSON exports to YOLO format:

```bash
python prepare_detection_yolo_dataset.py \
  -f annotations.json \
  -l annotations_group \
  --background \
  --compress
```

<details>
<summary><b>Full Arguments</b></summary>

| Argument | Required | Description |
|----------|----------|-------------|
| `-f, --project-exported-file` | ✅ | Label Studio JSON export |
| `-l, --label-by` | ✅ | Parent group name containing labels |
| `-i, --images-dir` | | Local directory to copy images from (avoids S3 downloads) |
| `-e, --exclude-classes` | | Comma-separated classes to exclude |
| `--single-class` | | Treat all classes as single target (id 0) |
| `--background` | | Include background images (~10% of dataset) |
| `--hallucinations` | | Directory with hard negative images |
| `--compress` | | Enable image compression |
| `--compress-size` | | Max dimension (default: 1280) |
| `--compress-quality` | | JPEG quality 1-100 (default: 95) |

**Outputs:** `project_folder/` (train/val splits), `dataset.yaml`, `classes.txt`

</details>

### 2️⃣ Run Inference

High-speed prediction on image collections:

```bash
python predict_and_filter.py \
  -w best.pt \
  -s /path/to/images \
  -o output_dir \
  --batch 8 \
  --db predictions.db
```

<details>
<summary><b>Full Arguments</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `-w, --weights` | | Path to YOLO weights (required) |
| `-s, --source` | | Base directory to scan (required) |
| `-o, --output` | | Output directory (required) |
| `-p, --pattern` | | Glob pattern for subdirectories |
| `-c, --conf` | 0.25 | Confidence threshold |
| `--imgsz` | 1280 | Inference image size |
| `--batch` | 8 | Batch size |
| `--half` | | Enable FP16 inference |
| `--db` | | SQLite database path |
| `--copy` | | Copy detected images to output |
| `--visualize` | | Save annotated images |
| `--select-random N` | | Sample N random images (instant indexing) |
| `--reindex` | | Force rebuild of image index |

**Outputs:** `output_dir/` (detections), `output_dir_visualized/` (annotations), `predictions.db`

</details>

### 3️⃣ Prediction API

FastAPI server for Label Studio integration:

```bash
python prediction_api.py \
  -w best.pt \
  -m "model_v1.0" \
  -H 0.0.0.0 \
  -s 8000
```

<details>
<summary><b>Arguments</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `-w, --weights` | | Path/URL to weights (required) |
| `-m, --model-version` | | Model name/version for tracking |
| `-d, --image-dir` | | Local image directory (avoids re-downloads) |
| `-H, --host` | 0.0.0.0 | API host |
| `-s, --port` | 8000 | API port |

**Endpoint:** `POST /predict` with JSON body:
```json
{"task": {...}, "project": 123}
```

</details>

### 4️⃣ Send Predictions to Label Studio

Automatically generate predictions for unlabeled tasks:

```bash
python send_prediction_tasks.py
```

<details>
<summary><b>Configuration</b></summary>

Edit environment variables in the script:
- `LABEL_STUDIO_TOKEN` – API token
- `LABEL_STUDIO_BASE_URL` – Label Studio URL

Fetches all tasks without annotations/predictions and sends them to your prediction API.

</details>

### 5️⃣ Export Annotations

Shell script to export Label Studio project:

```bash
bash export_annotations.sh
```

## 🏗️ Project Structure

```
dog_detection/
├── prepare_detection_yolo_dataset.py  # Dataset prep
├── predict_and_filter.py              # Batch inference
├── prediction_api.py                  # FastAPI server
├── send_prediction_tasks.py           # Label Studio integration
├── export_annotations.sh              # Export helper
├── requirements.txt
└── README.md
```

## 📊 Database Schema

<details>
<summary><b>Expand to view</b></summary>

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

</details>


## 📄 License

MIT License – see [LICENSE](LICENSE) file
