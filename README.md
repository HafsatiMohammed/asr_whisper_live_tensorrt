# WhisperLive with TensorRT Optimization on Jetson Platforms

## Introduction

This project aims to implement and optimize WhisperLive using TensorRT on Jetson platforms, including Orin (tested and works) and Xavier (to be tested). The primary goal is to overcome the issue with the `tensorrt_llm` library, which cannot be installed on Jetson platforms, by using `torch2trt` for model transformation. The server part will run inside a Docker container to ensure smooth operation and dependencies management.

## Features

- TensorRT optimization for WhisperLive on Jetson platforms
- Docker containerization for easy setup and deployment
- Support for multilingual models with options for translation and transcription

## Technology Stack

- Jetson Platforms (Orin, Xavier)
- TensorRT
- Docker
- torch2trt
- WhisperLive

## Installation and Setup

### Prerequisites

Ensure you have the necessary hardware and software requirements:
- Jetson platform (Orin or Xavier)
- Docker installed on your Jetson device

### Steps

1. **Clone this repository:**

    ```bash
    git clone https://gitlab.com/enchantedtools/interaction/asr_whisper_live_trt.git
    ```


2. **Clone the jetson-containers Repository:**

    ```bash
    git clone https://github.com/dusty-nv/jetson-containers
    ```

3. **Install jetson-containers:**

    ```bash
    bash jetson-containers/install.sh
    ```

4. **Copy Files:**

    Navigate to the `jetson-containers/packages/audio/whisper_trt/` directory and copy the necessary files from the cloned repository (the files in asr_whisper_live_trt) to this path.

5. **Build the Docker Container:**

    From any directory, run the following command to build the Docker container:

    ```bash
    jetson-containers run $(autotag whisper_trt)
    ```

    This process takes several minutes. You may encounter a test error at the end; this is not a problem.

6. **Verify Docker Image:**

    Check if the Docker image was built successfully:

    ```bash
    docker images
    ```

    Ensure the following line exists:

    ```plaintext
    whisper_trt                       r36.2.0-whisper_trt       0b62119606a3   4 hours ago    27.5GB
    ```

## Usage

### Running the Docker Container

You can run the Docker container using one of the following commands:

- **Option 1:**

    ```bash
    docker run --runtime nvidia -it --rm --network host \
        --volume /tmp/argus_socket:/tmp/argus_socket \
        --volume /etc/enctune.conf:/etc/enctune.conf \
        --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
        #--volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \ 
        --volume /var/run/dbus:/var/run/dbus \
        --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume /home/enchanted/Documents/jetson-containers/data:/data \
        --device /dev/snd --device /dev/bus/usb \
        -e DISPLAY=:1 -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v /tmp/.docker.xauth:/tmp/.docker.xauth \
        -e XAUTHORITY=/tmp/.docker.xauth \
        --device /dev/i2c-0 --device /dev/i2c-1 \
        --device /dev/i2c-2 --device /dev/i2c-3 \
        --device /dev/i2c-4 --device /dev/i2c-5 \
        --device /dev/i2c-6 --device /dev/i2c-7 \
        --device /dev/i2c-8 --device /dev/i2c-9 \
        -v /run/jtop.sock:/run/jtop.sock \
        whisper_trt
    ```

- **Option 2:**

    ```bash
    jetson-containers run $(autotag whisper_trt)
    ```

### Launching the Server Inside the Docker

To launch the server with TensorRT optimizations inside the Docker container, use:

```bash
python3 /opt/run_server.py -p 9090 -b tensorrt
```

### Running the Client

Outside the Docker container, you can launch the client using the following Python commands:

```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient("localhost", 9090, lang="fr", translate=False, model="large_v3")

# To transcribe a file
client(file_path)

# To stream from the microphone
client()
```

If needed, modify the WhisperLive client script to handle a specific microphone. Note that only multilingual models are supported, and you can specify the language and task (translation or transcription). Translation outputs are in English if another language is selected.

### Pull a Prebuilt Image and have all the models optimized only once

To pull the Docker image `hafsatimohammed/whisper_trt_with_files` from Docker Hub, use the following command:

```bash
docker pull hafsatimohammed/whisper_trt_with_files
```
To run the container and ensure that your models are stored, allowing you to go through the laborious optimization only once, you need to mount the volume where you want to store the models. Use the following command to run the container:

```bash
docker run --runtime nvidia -it --rm --network host \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --volume /etc/enctune.conf:/etc/enctune.conf \
  --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
  --volume /var/run/dbus:/var/run/dbus \
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume /home/enchanted/Documents/jetson-containers/data:/data \
  --volume /home/enchanted/Documents/jetson-containers/data/models/whisper:/root/.cache/whisper \
  hafsatimohammed/whisper_trt_with_files /bin/bash
```
In this example, the models will be stored in /home/enchanted/Documents/jetson-containers/data/models/whisper. Change the path to where you would like to store your models.

Once inside the Docker container, modify the main in model.py to download and optimize all the models you want (tiny, small, base, medium, large_v1, large_v2, or large_v3).

For instance, if you want to optimize the models with French as the language, it is used just for the tokenizer and will optimize the multilingual model. The language setting does not influence the optimization process.

```bash
docker run --runtime nvidia --rm --network host \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --volume /etc/enctune.conf:/etc/enctune.conf \
  --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
  --volume /var/run/dbus:/var/run/dbus \
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume /home/enchanted/Documents/jetson-containers/data:/data \
  --volume /home/enchanted/Documents/jetson-containers/data/models/whisper:/root/.cache/whisper \
  --entrypoint python3 hafsatimohammed/whisper_trt_with_files \
  /opt/run_server.py -p 9090 -b tensorrt
 ```

All you need to do is runing the client: 

In case you are in yocto or dont have the ability to install libraries: 
you don't have to install whisper_live outside the docker. All you need are the files in this repository to load: utils and client (in client you can comment line 16 and have utils imported instead)



## Contact

For any questions or issues, please contact Mohammed HAFSATI at mohammed@enchanted.tools



