#!/bin/bash
#SBATCH --job-name=O2_Prediction
#SBATCH --partition=hi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 # WARNING - hoping to increase these after imputation
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=logs/o2_pred_%j.out
#SBATCH --error=logs/o2_pred_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU information
lscpu

# Set threading limits # WARNING - hoping to increase these after imputation
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=4
ulimit -s unlimited
ulimit -u 1029238

# Set matplotlib to non-interactive backend
export MPLBACKEND=Agg

# Path to your Singularity container
CONTAINER_PATH="/home/hlsblun2/COVID_O2_HPC/O2_Prediction.sif"

# Path to your project directory (where your Python scripts are located)
PROJECT_DIR="/home/hlsblun2/COVID_O2_HPC"

# Change to project directory
cd $PROJECT_DIR

# Load module
module load singularity

# Check system limits
echo "3. Current system limits:"
echo "Max user processes: $(ulimit -u)"
echo "Stack size (KB): $(ulimit -s)"
echo "Max threads per process: $(cat /proc/sys/kernel/threads-max 2>/dev/null || echo 'unavailable')"

# Run the main pipeline script using Singularity with GPU support
echo "Starting O2_Prediction pipeline..."
singularity exec --bind $PROJECT_DIR:/workspace \
    $CONTAINER_PATH \
    python /workspace/pipeline.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully. Check model_outputs for the run results."
    echo "End Time: $(date)"
else
    echo "Pipeline failed with exit code $?"
    echo "End Time: $(date)"
    exit 1
fi

echo "Job completed at $(date)"