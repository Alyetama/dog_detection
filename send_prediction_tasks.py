#!/usr/bin/env python
# coding: utf-8

import os

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Update these three variables with your details
load_dotenv()
LS_URL = "https://label.biodiv.app"  # e.g., "http://localhost:8080"
API_KEY = os.environ['LABEL_STUDIO_TOKEN']
PROJECT_ID = 12

# Pointing directly to your YOLO prediction endpoint
ML_API_URL = "http://100.75.141.54:62000/predict"

HEADERS = {
    "Authorization": f"Token {API_KEY}",
    "Content-Type": "application/json"
}
# ==========================================


def get_tasks():
    """Fetches all tasks using the export endpoint to ensure we get annotations/predictions."""
    tqdm.write(f"Fetching data for project {PROJECT_ID}...")
    export_url = f"{LS_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"

    response = requests.get(export_url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def main():
    tasks = get_tasks()

    # Filter for tasks with NO annotations AND NO predictions
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

    # Process each task with a progress bar
    for task in tqdm(target_tasks, desc="Generating Predictions", unit="task"):
        task_id = task["id"]

        # Payload formatted exactly how your FastAPI script expects it
        payload = {"task": task, "project": PROJECT_ID}

        try:
            # 1. Ask your YOLO API for the prediction
            api_resp = requests.post(ML_API_URL, json=payload)

            if api_resp.status_code != 200:
                tqdm.write(
                    f"API Error on task {task_id}: {api_resp.status_code} - {api_resp.text}"
                )
                continue

            pred_data = api_resp.json()

            # Skip if the API returned an empty dict (no detections)
            if not pred_data or not pred_data.get("result"):
                continue

            # 2. Add the specific task ID so Label Studio knows where to attach it
            pred_data["task"] = task_id

            # 3. Push the finalized prediction dict into Label Studio
            ls_pred_url = f"{LS_URL}/api/predictions"
            ls_resp = requests.post(ls_pred_url,
                                    headers=HEADERS,
                                    json=pred_data)

            if ls_resp.status_code not in (200, 201):
                tqdm.write(
                    f"Label Studio Error on task {task_id}: {ls_resp.text}")

        except Exception as e:
            # Using tqdm.write prevents the print statement from breaking the visual progress bar
            tqdm.write(f"Exception processing task {task_id}: {e}")


if __name__ == "__main__":
    main()
