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
        self.api_endpoint = os.environ['API_ENDPOINT_URL'] + '/predict'

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            data = {'task': task, 'project': task.get('project')}

            try:
                r = requests.post(self.api_endpoint, json=data)
            except HTTPError as e:
                logger.error(f"HTTPError: {e}")
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

            if not pred:
                predictions.append({"result": []})
            else:
                predictions.append(pred)

        return predictions
