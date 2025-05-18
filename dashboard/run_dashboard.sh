#!/bin/bash
# Exo-Detector Dashboard Startup Script
# Author: Manus AI
# Date: May 2025

# Set default values
PORT=8501
DEBUG=false
DATA_DIR="$(pwd)/data"

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -p, --port PORT       Set port for Streamlit (default: 8501)"
    echo "  -d, --debug           Enable debug mode"
    echo "  --data-dir DIR        Set data directory (default: ./data)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with default settings"
    echo "  $0 -p 8502            # Run on port 8502"
    echo "  $0 -d                 # Run in debug mode"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --data-dir)
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

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory does not exist: $DATA_DIR"
    echo "Creating data directory..."
    mkdir -p "$DATA_DIR"
fi

# Set environment variables
export EXO_DETECTOR_DATA_DIR="$DATA_DIR"

# Set debug flag if needed
if [ "$DEBUG" = true ]; then
    export STREAMLIT_DEBUG=true
    echo "Debug mode enabled"
fi

# Run Streamlit
echo "Starting Exo-Detector Dashboard on port $PORT..."
streamlit run dashboard/app.py --server.port=$PORT
