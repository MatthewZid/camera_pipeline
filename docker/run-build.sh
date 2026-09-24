#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

debug="true"

if [ "$debug" = "true" ]; then
    docker build --target dev -t rosyolo -f docker/Dockerfile.cam .
else
    docker build --target production -t rosyolo -f docker/Dockerfile.cam .
fi