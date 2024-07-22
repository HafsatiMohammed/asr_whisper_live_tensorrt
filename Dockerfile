#---
# name: whisper_trt
# group: audio
# depends: [pytorch, torch2trt, onnxruntime]
# requires: '>=36'
# test: test.py
# notes: TensorRT optimized Whisper ASR from https://github.com/NVIDIA-AI-IOT/whisper_trt
#---



# Stage 1: Build base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS base


# Set the working directory
#WORKDIR /home/enchanted/Documents/Whisper_docker_tensorrt/

# Add versioning information
ADD https://api.github.com/repos/NVIDIA-AI-IOT/whisper_trt/git/refs/heads/main /tmp/whisper_trt_version.json

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
RUN pip3 install --no-cache-dir --verbose openai-whisper whisper-live librosa

# Clone whisper_trt repository and setup
RUN git clone https://github.com/NVIDIA-AI-IOT/whisper_trt /opt/whisper_trt && \
    cd /opt/whisper_trt && \
    pip3 install -e . && \
    mkdir -p ~/.cache && \
    ln -s /data/models/whisper ~/.cache/whisper && \
    ln -s /data/models/whisper ~/.cache/whisper_trt

# Stage 2: Final stage
FROM ${BASE_IMAGE}

# Copy the built files from the previous stage
COPY --from=base / /

# Copy modified files
COPY server.py /usr/local/lib/python3.10/dist-packages/whisper_live/server.py
COPY model.py /opt/whisper_trt/whisper_trt/model.py
COPY torch2trt.py /usr/local/lib/python3.10/dist-packages/torch2trt/torch2trt.py
COPY tensorrt_utils.py /usr/local/lib/python3.10/dist-packages/whisper_live/tensorrt_utils.py
COPY transcriber_tensorrt_whisper_trt.py /usr/local/lib/python3.10/dist-packages/whisper_live/transcriber_tensorrt_whisper_trt.py
COPY run_server.py /opt/
COPY vad.py /usr/local/lib/python3.10/dist-packages/whisper_live/vad.py
COPY assets /usr/local/lib/python3.10/dist-packages/whisper_live/
COPY assets ~/.cache/whisper/
COPY run_models.sh /usr/local/lib/python3.10/dist-packages/whisper_live/run_models.sh
COPY __init__.py   /usr/local/lib/python3.10/dist-packages/whisper/__init__.py 
#RUN chmod +x /usr/local/lib/python3.10/dist-packages/whisper_live/run_models.sh
#Run chmod +x /opt/whisper_trt/whisper_trt/model.py
#Run python3 /opt/whisper_trt/whisper_trt/model.py
#RUN bash /usr/local/lib/python3.10/dist-packages/whisper_live/run_models.sh


# Set the entrypoint or command if necessary
# CMD ["python3", "/opt/run_server.py"]
