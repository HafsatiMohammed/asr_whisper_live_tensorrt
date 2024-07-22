#!/bin/bash

# Ensure the /root/.cache/whisper directory exists
mkdir -p /root/.cache/whisper

# Run the Python script
python3 /opt/whisper_trt/whisper_trt/model.py