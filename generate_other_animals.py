#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import boto3
from dotenv import load_dotenv
from tqdm import tqdm


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect images labeled 'other animal' into a local folder.")
    parser.add_argument(
        '-f', '--project-exported-file',
        type=str,
        required=True,
        help='Exported JSON MIN file from Label Studio')
    parser.add_argument(
        '-l', '--label-by',
        type=str,
        default='label',
        help='The parent group of labels used in annotations (default: label)')
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='other_animals',
        help='Destination folder for collected images (default: other_animals)')
    parser.add_argument(
        '-i', '--images-dir',
        type=str,
        default=None,
        help='Local directory to copy images from to avoid downloading')
    parser.add_argument(
        '-c', '--class-name',
        type=str,
        default='other animal',
        help='Class name to collect (default: "other animal")')
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = opts()

    import json
    with open(args.project_exported_file) as f:
        data = json.load(f)

    matching_tasks = []
    for task in data:
        for ann in task.get(args.label_by, []):
            if args.class_name in ann.get('rectanglelabels', []):
                matching_tasks.append(task)
                break

    print(f"Found {len(matching_tasks)} images labeled '{args.class_name}'.")

    if not matching_tasks:
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    s3_client = None

    for task in tqdm(matching_tasks, desc="Collecting images"):
        image_filename = Path(task['image']).name
        dest = output_path / image_filename

        if dest.exists():
            continue

        copied = False
        if args.images_dir:
            src = Path(args.images_dir) / image_filename
            if src.exists():
                shutil.copy(src, dest)
                copied = True

        if not copied:
            if s3_client is None:
                s3_client = boto3.client(
                    's3',
                    endpoint_url=os.getenv('ENDPOINT_URL'),
                    aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
                    region_name=os.getenv('BUCKET_REGION'))

            s3_key = '/'.join(Path(task['image']).parts[2:])
            s3_client.download_file(os.getenv('BUCKET_NAME'), s3_key, str(dest))

    print(f"Done. Images saved to '{args.output_dir}/'.")


if __name__ == '__main__':
    main()
