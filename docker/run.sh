#!/bin/bash
# Exo-Detector Docker Build and Run Script
# Author: Manus AI
# Date: May 2025

# Set default values
IMAGE_NAME="exo-detector"
TAG="latest"
USE_GPU=false
COMMAND="src/run_phase1.py"
MOUNT_DATA=true
DATA_DIR="$(pwd)/data"

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -g, --gpu                 Use GPU-enabled Docker image"
    echo "  -i, --image NAME          Set Docker image name (default: exo-detector)"
    echo "  -t, --tag TAG             Set Docker image tag (default: latest)"
    echo "  -c, --command COMMAND     Set command to run (default: src/run_phase1.py)"
    echo "  --no-mount                Don't mount data directory"
    echo "  -d, --data-dir DIR        Set data directory to mount (default: ./data)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu -c src/run_phase2.py    # Run Phase 2 with GPU"
    echo "  $0 -c dashboard/app.py           # Run Streamlit dashboard"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -g|--gpu)
            USE_GPU=true
            shift
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -t|--tag)
            TAG="$2"
            shift
            shift
            ;;
        -c|--command)
            COMMAND="$2"
            shift
            shift
            ;;
        --no-mount)
            MOUNT_DATA=false
            shift
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set Dockerfile based on GPU flag
if [ "$USE_GPU" = true ]; then
    DOCKERFILE="docker/Dockerfile.gpu"
    DOCKER_BUILD_ARGS="--build-arg CUDA_VERSION=11.7.0-cudnn8-runtime-ubuntu20.04"
    DOCKER_RUN_ARGS="--gpus all"
    IMAGE_TAG="${IMAGE_NAME}:${TAG}-gpu"
else
    DOCKERFILE="docker/Dockerfile.cpu"
    DOCKER_BUILD_ARGS=""
    DOCKER_RUN_ARGS=""
    IMAGE_TAG="${IMAGE_NAME}:${TAG}"
fi

# Build Docker image
echo "Building Docker image: $IMAGE_TAG"
echo "Using Dockerfile: $DOCKERFILE"

docker build $DOCKER_BUILD_ARGS -t $IMAGE_TAG -f $DOCKERFILE .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Prepare run command
RUN_CMD="docker run --rm -it"

# Add volume mount if requested
if [ "$MOUNT_DATA" = true ]; then
    echo "Mounting data directory: $DATA_DIR"
    RUN_CMD="$RUN_CMD -v $DATA_DIR:/app/data"
fi

# Add GPU args if needed
if [ "$USE_GPU" = true ]; then
    RUN_CMD="$RUN_CMD $DOCKER_RUN_ARGS"
fi

# Add port mapping if running dashboard
if [[ "$COMMAND" == *"dashboard/app.py"* ]]; then
    RUN_CMD="$RUN_CMD -p 8501:8501"
    COMMAND="$COMMAND --server.port=8501 --server.address=0.0.0.0"
fi

# Complete the command
RUN_CMD="$RUN_CMD $IMAGE_TAG $COMMAND"

# Run the container
echo "Running command: $RUN_CMD"
eval $RUN_CMD
