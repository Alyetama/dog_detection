#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import re
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None


def extract_region_info(file_path_str: str):
    """Extracts image ID, region, and parent region based on folder structure."""
    p = Path(file_path_str)
    image_id = p.stem
    region_name = p.parent.parent.name
    parent_region_name = re.sub(r'(_-?\d+(\.\d+)?)+$', '', region_name)
    return image_id, region_name, parent_region_name


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "High-Speed YOLO scanner with SQLite logging and multithreaded I/O.")
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='Path to weights (e.g., best.pt)')
    parser.add_argument('-s',
                        '--source',
                        type=str,
                        required=True,
                        help='Base directory to scan')
    parser.add_argument('-p',
                        '--pattern',
                        type=str,
                        default=None,
                        help='Optional glob pattern')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Output directory')
    parser.add_argument('-c',
                        '--conf',
                        type=float,
                        default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--imgsz',
                        type=int,
                        default=1280,
                        help='Inference image size')
    parser.add_argument('--db',
                        type=str,
                        default=None,
                        help='Path to SQLite database')
    parser.add_argument('--batch',
                        type=int,
                        default=8,
                        help='Batch size for GPU inference')
    parser.add_argument('--half',
                        action='store_true',
                        help='Enable FP16 half-precision inference')
    parser.add_argument(
        '--copy',
        action='store_true',
        help=
        'Enable copying of detected original images to the output directory')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help=
        'Save images with drawn bounding boxes to a separate _visualized folder'
    )
    parser.add_argument(
        '--select-random',
        type=int,
        default=None,
        help='Randomly select N unprocessed images across regions instantly.')

    # --- NEW ARGUMENT ---
    parser.add_argument(
        '--reindex',
        action='store_true',
        help=
        'Force rebuild of the image index text file if you added new images.')

    return parser.parse_args()


def get_target_directories(source_dir: str, pattern: str) -> list:
    base_path = Path(source_dir)
    if pattern:
        return [d for d in base_path.glob(pattern) if d.is_dir()]
    else:
        return [base_path] if base_path.is_dir() else []


def setup_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region_name TEXT,
            parent_region_name TEXT,
            image_full_path TEXT UNIQUE,
            image_id TEXT,
            num_detections INTEGER,
            conf_per_detection TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_full_path TEXT UNIQUE,
            image_id TEXT,
            region_name TEXT,
            parent_region_name TEXT,
            error_message TEXT
        )
    ''')
    conn.commit()
    return conn, cursor


def get_processed_state(cursor) -> dict:
    if not cursor:
        return {}
    processed = {}
    cursor.execute("SELECT image_full_path, num_detections FROM predictions")
    for row in cursor.fetchall():
        processed[row[0]] = row[1]
    cursor.execute("SELECT image_full_path FROM errors")
    for row in cursor.fetchall():
        processed[row[0]] = -1
    return processed


def get_missing_copies(processed: dict, output_dir: str,
                       do_copy: bool) -> list:
    """Instantly checks the database state to find images that need back-copying."""
    needs_copy = []
    if do_copy:
        for full_path, num_det in processed.items():
            if num_det is not None and num_det > 0:
                filename = Path(full_path).name
                dest_path = os.path.join(output_dir, filename)
                if not os.path.exists(dest_path):
                    needs_copy.append((full_path, dest_path))
    return needs_copy


def get_unprocessed_files(directories: list, processed: dict, output_dir: str,
                          do_copy: bool) -> tuple:
    """Standard exhaustive search."""
    valid_extensions = {
        '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'
    }
    dir_to_files = {}
    total_count = 0

    for d in directories:
        files_in_dir = []
        try:
            for entry in os.scandir(d):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in valid_extensions:
                        if entry.path not in processed:
                            files_in_dir.append(entry.path)
        except (PermissionError, FileNotFoundError):
            pass

        if files_in_dir:
            dir_to_files[d] = files_in_dir
            total_count += len(files_in_dir)

    needs_copy = get_missing_copies(processed, output_dir, do_copy)
    return dir_to_files, total_count, needs_copy


def get_random_unprocessed_files(directories: list, processed: dict,
                                 output_dir: str, do_copy: bool,
                                 sample_size: int,
                                 force_reindex: bool) -> tuple:
    """High-speed index-based random sampling (Instant for millions of files)."""
    index_file = "dataset_image_index.txt"

    if force_reindex and os.path.exists(index_file):
        os.remove(index_file)

    # 1. Build Index if missing (Only happens once)
    if not os.path.exists(index_file):
        print(f"\n[INFO] Building dataset index for the first time...")
        print(
            "[INFO] This takes a few minutes, but makes all future random sampling INSTANT."
        )
        valid_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'
        }

        with open(index_file, 'w') as f:
            for d in tqdm(directories, desc="Indexing Folders", unit="dir"):
                try:
                    for entry in os.scandir(d):
                        if entry.is_file():
                            if os.path.splitext(
                                    entry.name)[1].lower() in valid_extensions:
                                f.write(entry.path + '\n')
                except (PermissionError, FileNotFoundError):
                    pass

    # 2. Load Index instantly into RAM
    print("Loading image index into memory...")
    with open(index_file, 'r') as f:
        all_files = f.read().splitlines()

    # 3. Filter out processed files and shuffle the rest
    unprocessed_pool = [f for f in all_files if f not in processed]
    take_count = min(sample_size, len(unprocessed_pool))

    if take_count == 0:
        return {}, 0, []

    sampled_files = random.sample(unprocessed_pool, take_count)

    # 4. Find past detections missing from output dir
    needs_copy = get_missing_copies(processed, output_dir, do_copy)

    dir_to_files = {"random_selection": sampled_files}
    return dir_to_files, len(sampled_files), needs_copy


def copy_file_worker(src: str, dest: str):
    try:
        shutil.copy(src, dest)
    except Exception:
        pass


def save_visualized_worker(annotated_img_array, dest_path: str):
    try:
        cv2.imwrite(dest_path, annotated_img_array)
    except Exception:
        pass


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = opts()

    os.makedirs(args.output, exist_ok=True)
    vis_output_dir = f"{args.output}_visualized"
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)

    conn, cursor = None, None
    processed_state = {}

    if args.db:
        print(f"Setting up database at {args.db}...")
        conn, cursor = setup_database(args.db)
        processed_state = get_processed_state(cursor)
        if processed_state:
            print(
                f"Found {len(processed_state)} previously processed/errored images in the database."
            )

    target_dirs = get_target_directories(args.source, args.pattern)
    if not target_dirs:
        print("No valid directories found. Exiting.")
        if conn: conn.close()
        return

    # --- USE THE INSTANT INDEX STRATEGY ---
    if args.select_random:
        dir_to_files, total_images, needs_copy = get_random_unprocessed_files(
            target_dirs, processed_state, args.output, args.copy,
            args.select_random, args.reindex)
    else:
        dir_to_files, total_images, needs_copy = get_unprocessed_files(
            target_dirs, processed_state, args.output, args.copy)

    io_executor = ThreadPoolExecutor(max_workers=8)

    if needs_copy and args.copy:
        print(
            f"Found {len(needs_copy)} previously detected images missing from output dir. Copying them now..."
        )
        for src, dest in needs_copy:
            io_executor.submit(copy_file_worker, src, dest)

    if total_images == 0:
        print("No new valid images found to run predictions on. Exiting.")
        io_executor.shutdown(wait=True)
        if conn: conn.close()
        return

    print(f"\nLoading model from {args.weights}...")
    model = YOLO(args.weights)

    print(f"Scanning {total_images} new images...")
    print(f"Batch Size: {args.batch} | Half-Precision: {args.half}")
    print(
        f"Copy Originals: {args.copy} | Save Visualizations: {args.visualize}")
    print("-" * 50)

    copied_count = 0
    total_processed = 0
    total_detections = 0

    progress_bar = tqdm(total=total_images,
                        desc="Scanning",
                        unit="img",
                        smoothing=0.1)

    for target_dir, files in dir_to_files.items():
        for file_chunk in chunk_list(files, 5000):

            temp_txt_path = f"temp_yolo_source_{os.getpid()}.txt"
            with open(temp_txt_path, "w") as f:
                f.write("\n".join(file_chunk))

            try:
                results = model.predict(source=temp_txt_path,
                                        conf=args.conf,
                                        imgsz=args.imgsz,
                                        batch=args.batch,
                                        half=args.half,
                                        stream=True,
                                        verbose=False)

                processed_in_chunk = set()

                for result in results:
                    total_processed += 1
                    img_path = result.path
                    processed_in_chunk.add(Path(img_path).name)

                    image_id, region_name, parent_region_name = extract_region_info(
                        img_path)
                    filename = Path(img_path).name
                    num_boxes = len(result.boxes)

                    if num_boxes > 0:
                        num_insert = num_boxes
                        confs = result.boxes.conf.cpu().numpy()
                        conf_str = ",".join([f"{c:.3f}" for c in confs])
                        has_detection = True
                        total_detections += num_boxes
                    else:
                        num_insert = None
                        conf_str = None
                        has_detection = False

                    if cursor:
                        cursor.execute(
                            '''
                            INSERT OR REPLACE INTO predictions 
                            (region_name, parent_region_name, image_full_path, image_id, num_detections, conf_per_detection)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (region_name, parent_region_name, img_path,
                              image_id, num_insert, conf_str))

                        if total_processed % 1000 == 0:
                            conn.commit()

                    if has_detection:
                        if args.copy:
                            dest_path = os.path.join(args.output, filename)
                            io_executor.submit(copy_file_worker, img_path,
                                               dest_path)
                            copied_count += 1

                        if args.visualize:
                            vis_dest_path = os.path.join(
                                vis_output_dir, filename)
                            annotated_frame = result.plot(labels=False,
                                                          conf=True)
                            io_executor.submit(save_visualized_worker,
                                               annotated_frame, vis_dest_path)

                    progress_bar.update(1)
                    progress_bar.set_postfix(detections=total_detections)

                for chunk_img_path in file_chunk:
                    if Path(chunk_img_path).name not in processed_in_chunk:
                        err_id, err_region, err_parent = extract_region_info(
                            chunk_img_path)
                        if cursor:
                            cursor.execute(
                                '''
                                INSERT OR REPLACE INTO errors 
                                (image_full_path, image_id, region_name, parent_region_name, error_message)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (chunk_img_path, err_id, err_region,
                                  err_parent,
                                  "Skipped by YOLO (corrupt/unreadable)"))

                        progress_bar.update(1)
                        total_processed += 1

            finally:
                if os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

    progress_bar.close()

    print(
        "Inference complete. Waiting for final background file copies/saves to finish..."
    )
    io_executor.shutdown(wait=True)

    if conn:
        conn.commit()
        conn.close()

    print("\n" + "=" * 40)
    print("FINISHED")
    print(f"Total new images scanned: {total_processed}")
    print(f"Total objects detected: {total_detections}")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
