#!/usr/bin/env python

import argparse
import os

import requests
from dotenv import load_dotenv
from tqdm import tqdm


def get_tasks(ls_url, project_id, headers):
    """Fetches all tasks using the export endpoint to ensure we get annotations/predictions."""
    tqdm.write(f"Fetching data for project {project_id}...")
    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true"

    response = requests.get(export_url, headers=headers)
    response.raise_for_status()
    return response.json()


def main(args, api_key):
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    tasks = get_tasks(args.ls_url, args.project_id, headers)

    target_tasks = []
    for task in tasks:
        has_annotations = len(task.get("annotations", [])) > 0
        has_predictions = len(task.get("predictions", [])) > 0

        if not has_annotations and not has_predictions:
            target_tasks.append(task)

    tqdm.write(
        f"Found {len(target_tasks)} tasks requiring predictions out of {len(tasks)} total."
    )

    if not target_tasks:
        tqdm.write("All tasks are annotated or have predictions. Exiting.")
        return

    for task in tqdm(target_tasks, desc="Generating Predictions", unit="task"):
        task_id = task["id"]
        payload = {"task": task, "project": args.project_id}

        try:
            api_resp = requests.post(args.ml_api_url, json=payload)

            if api_resp.status_code != 200:
                tqdm.write(
                    f"API Error on task {task_id}: {api_resp.status_code} - {api_resp.text}"
                )
                continue

            pred_data = api_resp.json()

            if not pred_data or not pred_data.get("result"):
                continue

            pred_data["task"] = task_id

            ls_pred_url = f"{args.ls_url}/api/predictions"
            ls_resp = requests.post(ls_pred_url,
                                    headers=headers,
                                    json=pred_data)

            if ls_resp.status_code not in (200, 201):
                tqdm.write(
                    f"Label Studio Error on task {task_id}: {ls_resp.text}")

        except Exception as e:
            tqdm.write(f"Exception processing task {task_id}: {e}")


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ.get("LABEL_STUDIO_TOKEN")
    if not api_key:
        raise ValueError("LABEL_STUDIO_TOKEN must be set in the .env file")

    parser = argparse.ArgumentParser(
        description=
        "Send unannotated tasks to a YOLO ML endpoint for predictions.")
    parser.add_argument("--ls-url",
                        type=str,
                        required=True,
                        help="Label Studio instance URL")
    parser.add_argument("--project-id",
                        type=int,
                        required=True,
                        help="Label Studio Project ID")
    parser.add_argument("--ml-api-url",
                        type=str,
                        required=True,
                        help="ML API Endpoint URL")

    args = parser.parse_args()

    main(args, api_key)
