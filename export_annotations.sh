#!/bin/bash

current_date=$(date +"%Y-%m-%d-%H-%M")
random_string=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

set -a
[ -f .env ] && source .env
set +a

TOKEN=$LABEL_STUDIO_TOKEN

curl -X GET \
    "https://label.biodiv.app/api/projects/12/export?exportType=JSON_MIN" \
    -H "Authorization: Token $TOKEN" \
    --output  "project-12-at-$current_date-$random_string.json"

echo "project-12-at-$current_date-$random_string.json"
