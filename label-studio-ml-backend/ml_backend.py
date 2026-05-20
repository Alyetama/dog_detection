#!/usr/bin/env python
# coding: utf-8

import os

import requests
from dotenv import load_dotenv
from label_studio_ml.model import LabelStudioMLBase
from loguru import logger
from requests.exceptions import HTTPError

load_dotenv()


class MyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Verify this port matches where you are running api_md_latest.py
        self.api_endpoint = os.environ['API_ENDPOINT_URL'] + '/predict'

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            data = {'task': task, 'project': task.get('project')}

            try:
                r = requests.post(self.api_endpoint, json=data)
            except HTTPError as e:
                logger.error(f"HTTPError: {e}")
                # Append empty results so Label Studio doesn't crash on this task
                predictions.append({"result": []})
                continue
            except Exception as e:
                logger.error(f"Exception: {e}")
                predictions.append({"result": []})
                continue

            if r.status_code != 200:
                logger.error(f"API Error ({r.status_code}): {r.text}")
                predictions.append({"result": []})
                continue

            pred = r.json()

            # If the API returns {} (no detections), give Label Studio the expected format
            if not pred:
                predictions.append({"result": []})
            else:
                predictions.append(pred)

        # Return the full list of predictions for all tasks after the loop finishes
        return predictions
