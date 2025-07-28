#!/bin/bash

# Build script for O2_Prediction Singularity container with GPU support

set -e  # Exit on any error

echo "Building O2_Prediction Singularity container with GPU support..."

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    echo "Error: Singularity is not installed or not in PATH"
    exit 1
fi

# Build the container
echo "Starting container build process..."
singularity build O2_Prediction.sif O2_Prediction.def

if [ $? -eq 0 ]; then
    echo "Container built successfully!"
    echo "Container file: O2_Prediction.sif"
    echo "Size: $(ls -lh O2_Prediction.sif | awk '{print $5}')"

    # Test the container
    echo "Testing the container..."
    singularity exec O2_Prediction.sif python -c "
import torch
import numpy, pandas, sklearn, xgboost, mlflow, shap, statsmodels
print('All packages working correctly')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
"

    if [ $? -eq 0 ]; then
        echo "Container test passed"
    else
        echo "Container test failed"
        exit 1
    fi
else
    echo "Container build failed"
    exit 1
fi

echo "Build process completed successfully"
