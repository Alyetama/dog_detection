#!/usr/bin/env python

import argparse
import json
import os
import random
import shutil
from glob import glob
from pathlib import Path

import boto3
import cv2
from dotenv import load_dotenv
from tqdm import tqdm


def bbox_ls_to_yolo(x: float, y: float, width: float, height: float) -> tuple:
    """
    Converts Label Studio bounding box coordinates to YOLO format.
    """
    x = (x + width / 2) / 100
    y = (y + height / 2) / 100
    w = width / 100
    h = height / 100
    return x, y, w, h


def create_label_files(task: dict, labels_dest: str, label_by: str,
                       class_mapping: dict) -> None:
    """
    Parses a single Label Studio task and generates the corresponding YOLO .txt label file.
    """
    image_filename = Path(task["image"]).stem
    label_file_dest = f'{labels_dest}/{image_filename}.txt'

    if task.get('background') == 'yes':
        with open(label_file_dest, 'w') as f:
            pass
        return

    lines = []
    valid_annotations_found = False

    for ann in task.get(label_by, []):
        if not all(k in ann
                   for k in ('x', 'y', 'width',
                             'height')) or not ann.get('rectanglelabels'):
            continue

        class_name = ann['rectanglelabels'][0]

        if class_name not in class_mapping:
            continue

        valid_annotations_found = True
        yolo_bbox = bbox_ls_to_yolo(ann['x'], ann['y'], ann['width'],
                                    ann['height'])
        class_id = class_mapping[class_name]

        line_data = [str(class_id)] + [str(coord) for coord in yolo_bbox]
        lines.append(' '.join(line_data))

    if valid_annotations_found:
        with open(label_file_dest, 'w') as f:
            f.write('\n'.join(lines) + '\n')


def split_data(output_dir: str,
               images_source_dir: str = 'ls_images',
               labels_source_dir: str = 'ls_labels',
               seed: int = 8) -> None:
    """
    Cleans up mismatched files and splits the dataset into an 80/20 train and validation set.
    """
    random.seed(seed)

    imgs_full = glob(f'{output_dir}/{images_source_dir}/*')
    imgs = [Path(x).stem for x in imgs_full]
    labels_full = glob(f'{output_dir}/{labels_source_dir}/*')
    labels = [Path(x).stem for x in labels_full]

    in_imgs_but_not_in_labels = [x for x in imgs if x not in labels]
    in_labels_but_not_in_images = [x for x in labels if x not in imgs]

    imgs_to_delete = [
        x for x in imgs_full if Path(x).stem in in_imgs_but_not_in_labels
    ]
    labels_to_delete = [
        x for x in labels_full if Path(x).stem in in_labels_but_not_in_images
    ]

    for item in imgs_to_delete + labels_to_delete:
        Path(item).unlink()

    for subdir in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        Path(f'{output_dir}/{subdir}').mkdir(parents=True, exist_ok=True)

    images = sorted(glob(f'{output_dir}/{images_source_dir}/*'))
    labels = sorted(glob(f'{output_dir}/{labels_source_dir}/*'))
    pairs = list(zip(images, labels))

    train_len = round(len(pairs) * 0.8)
    random.shuffle(pairs)

    train, val = pairs[:train_len], pairs[train_len:]

    for im, label in tqdm(train, desc="Splitting Train"):
        shutil.copy(im, f'{output_dir}/images/train')
        shutil.copy(label, f'{output_dir}/labels/train')

    for im, label in tqdm(val, desc="Splitting Val"):
        shutil.copy(im, f'{output_dir}/images/val')
        shutil.copy(label, f'{output_dir}/labels/val')

    shutil.rmtree(f'{output_dir}/{images_source_dir}', ignore_errors=True)
    shutil.rmtree(f'{output_dir}/{labels_source_dir}', ignore_errors=True)


def compress_image(image_path: str, max_dim: int, jpeg_quality: int) -> None:
    """
    Reads an image, resizes it if the longest side exceeds max_dim preserving aspect ratio, 
    and overwrites it with specified JPEG quality.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return

        h, w = img.shape[:2]

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(image_path, img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    except Exception as e:
        print(f"Failed to compress {image_path}: {e}")


def run(project_exported_file: str, label_by: str, images_dir: str,
        exclude_classes: list, single_class: bool, use_background: bool,
        hallucinations_dir: str, compress: bool, compress_size: int,
        compress_quality: int) -> None:
    """
    Main orchestration function to fetch images, parse labels, apply compression, and generate the YOLO dataset.
    """
    images_source_dir = 'ls_images'
    labels_source_dir = 'ls_labels'

    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('ENDPOINT_URL'),
        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        region_name=os.getenv('BUCKET_REGION'))

    project_folder = Path(project_exported_file).stem
    Path(f'{project_folder}/{images_source_dir}').mkdir(exist_ok=True,
                                                        parents=True)
    Path(f'{project_folder}/{labels_source_dir}').mkdir(exist_ok=True)

    with open(project_exported_file) as j:
        raw_data = json.load(j)

    object_tasks = [x for x in raw_data if x.get(label_by)]
    background_tasks = [x for x in raw_data if x.get('background') == 'yes']
    hal_tasks = []

    if hallucinations_dir:
        hal_path = Path(hallucinations_dir)
        if hal_path.exists() and hal_path.is_dir():
            for img_file in hal_path.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in [
                        '.jpg', '.jpeg', '.png'
                ]:
                    hal_tasks.append({
                        'image': str(img_file.absolute()),
                        'background': 'yes',
                        'is_hallucination': True
                    })
            print(
                f"Found {len(hal_tasks)} hard negative images in {hallucinations_dir}."
            )
        else:
            print(
                f"Warning: Hallucination directory '{hallucinations_dir}' not found or is not a directory."
            )

    names_set = set()
    for task in object_tasks:
        for ann in task.get(label_by, []):
            if ann.get('rectanglelabels'):
                for lbl in ann['rectanglelabels']:
                    if lbl not in exclude_classes:
                        names_set.add(lbl)

    if single_class:
        names = {"target": 0}
        class_mapping = {lbl: 0 for lbl in names_set}
        print(
            f"Single-class mode enabled. Grouping {list(names_set)} into 'target'."
        )
    else:
        names = {name: idx for idx, name in enumerate(sorted(list(names_set)))}
        class_mapping = names
        print(f"Keeping the following classes: {names}")

    if exclude_classes:
        print(f"Excluding the following classes: {exclude_classes}")

    if use_background or hal_tasks:
        max_bg_count = int(len(object_tasks) / 9)

        if not use_background:
            background_tasks = []

        if len(hal_tasks) > max_bg_count:
            print(
                f"Note: Capping hallucinations to {max_bg_count} to strictly maintain the 10% limit."
            )
            random.shuffle(hal_tasks)
            hal_tasks = hal_tasks[:max_bg_count]

        allowed_random_bgs = max(0, max_bg_count - len(hal_tasks))
        if len(background_tasks) > allowed_random_bgs:
            random.shuffle(background_tasks)
            background_tasks = background_tasks[:allowed_random_bgs]

        final_backgrounds = hal_tasks + background_tasks
        print(
            f"Injecting {len(hal_tasks)} hallucinations and {len(background_tasks)} random backgrounds."
        )
    else:
        final_backgrounds = []

    data = object_tasks + final_backgrounds

    bucket_name = os.getenv('BUCKET_NAME')

    for task in tqdm(data, desc="Fetching Images"):
        image_filename = Path(task['image']).name
        local_dest_path = Path(
            f'{project_folder}/{images_source_dir}/{image_filename}')

        if local_dest_path.exists():
            continue

        if task.get('is_hallucination'):
            shutil.copy(task['image'], local_dest_path)
        else:
            copied_from_local = False
            if images_dir:
                potential_source = Path(images_dir) / image_filename
                if potential_source.exists():
                    shutil.copy(potential_source, local_dest_path)
                    copied_from_local = True

            if not copied_from_local:
                s3_key = '/'.join(Path(task['image']).parts[2:])
                s3_client.download_file(bucket_name, s3_key,
                                        str(local_dest_path))

        if compress:
            compress_image(str(local_dest_path), compress_size,
                           compress_quality)

    labels_dest = f'{project_folder}/{labels_source_dir}'
    for task in tqdm(data, desc="Creating Labels"):
        create_label_files(task, labels_dest, label_by, class_mapping)

    split_data(project_folder)

    with open(f'{project_folder}/classes.txt', 'w') as f:
        for name, idx in names.items():
            f.write(f"{idx}: {name}\n")

    abs_project_folder = Path(project_folder).absolute()

    yaml_lines = [
        f"path: {abs_project_folder}", "train: images/train",
        "val: images/val", "", "names:"
    ]

    sorted_names = sorted(names.items(), key=lambda item: item[1])
    for name, idx in sorted_names:
        yaml_lines.append(f"  {idx}: {name}")

    with open("dataset.yaml", "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"\nSaved to project folder: {project_folder}")


def opts() -> argparse.Namespace:
    """
    Parses and returns command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--project-exported-file',
                        help='Exported JSON MIN file from label-studio',
                        type=str,
                        required=True)
    parser.add_argument('-l',
                        '--label-by',
                        help='The parent group of labels to use for detection',
                        required=True)
    parser.add_argument(
        '-i',
        '--images-dir',
        help='Local directory to copy images from to avoid downloading',
        type=str,
        default=None)
    parser.add_argument(
        '-e',
        '--exclude-classes',
        type=str,
        default="",
        help=
        'Comma-separated list of class names to exclude (e.g., -e "exclude,other animal")'
    )
    parser.add_argument(
        '--single-class',
        action='store_true',
        help='Treat all included classes as a single class (id 0)')
    parser.add_argument(
        '--background',
        action='store_true',
        help='Include empty background images up to 10%% of the dataset')
    parser.add_argument(
        '--hallucinations',
        type=str,
        default=None,
        help=
        'Path to a local directory containing images where the model falsely detected an object (hard negatives).'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Enable image compression and resizing during fetching.')
    parser.add_argument(
        '--compress-size',
        type=int,
        default=1280,
        help=
        'Maximum dimension for the longest side if compression is enabled (default: 1280).'
    )
    parser.add_argument(
        '--compress-quality',
        type=int,
        default=95,
        help=
        'JPEG compression quality 1-100 if compression is enabled (default: 95).'
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for execution: loads environment variables, parses arguments, and triggers the run sequence.
    """
    load_dotenv()
    args = opts()

    exclude_list = [c.strip() for c in args.exclude_classes.split(',')
                    ] if args.exclude_classes else []

    run(project_exported_file=args.project_exported_file,
        label_by=args.label_by,
        images_dir=args.images_dir,
        exclude_classes=exclude_list,
        single_class=args.single_class,
        use_background=args.background,
        hallucinations_dir=args.hallucinations,
        compress=args.compress,
        compress_size=args.compress_size,
        compress_quality=args.compress_quality)


if __name__ == '__main__':
    main()
