#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

debug="true"

if [ "$debug" = "true" ]; then
    docker compose up -d
else
    docker compose -f docker-compose.yml up -d
fi

docker logs -f camera_pipeline