#!/bin/bash

# User-specified date prefix (required)
DATE_PREFIX="${1:-260414}"

# Define model backends and their abbreviations
# Format: "backend name:abbreviation"
declare -a BACKENDS=(
    "naive:naive"
    "mobile:mobile"
    "pann:pann"
)

# Define datasets and their abbreviations
# Format: "dataset name:abbreviation"
declare -a DATASETS=(
    "cough-speech-sneeze:css"
    "CoughDataset:coughvid"
    "ESC50-human:esc50"
)

# Set seed list
seeds=(7 42 123 1309 5287 31415)
# seeds=(42)  # For quick testing

# Define all model types
types=(
    "Mel"
    "MelLDS"
    "MelHilbert"
    "MelHilbertLDS"
    "MelHilbertTime"
    "MelHilbertTimeLDS"
    "SignalHilbert"
    "SignalHilbertLDS"
    "Signal"
    "SignalLDS"
)

# Iterate over each model backend
for backend_info in "${BACKENDS[@]}"; do
    # Parse backend name and abbreviation
    IFS=':' read -r backend_name backend_abbr <<< "$backend_info"
    
    echo "========================================"
    echo "Using model backend: $backend_name (abbreviation: $backend_abbr)"
    echo "========================================"
    
    # Iterate over each dataset
    for dataset_info in "${DATASETS[@]}"; do
        # Parse dataset name and abbreviation
        IFS=':' read -r dataset_name dataset_abbr <<< "$dataset_info"
        
        # Auto-generate run_id
        run_id="${DATE_PREFIX}_${backend_abbr}_${dataset_abbr}"
        
        echo "----------------------------------------"
        echo "Dataset: $dataset_name (abbreviation: $dataset_abbr)"
        echo "Run ID: $run_id"
        echo "----------------------------------------"
        
        # Create runs directory
        mkdir -p "runs_${run_id}"
        
        # Run each model type sequentially
        for type in "${types[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Training: type=${type}, seed=${seed}, backend=${backend_name}, dataset=${dataset_name}"
                
                # Run training script
                python exp_css/train_css.py \
                    --type "${type}" \
                    --date "${run_id}" \
                    --seed "${seed}" \
                    --hop_length 64 \
                    --dataset "${dataset_name}" \
                    --model_backend "${backend_name}"
                
                # Check if training succeeded
                if [ $? -ne 0 ]; then
                    echo "Warning: Training failed - type=${type}, seed=${seed}, backend=${backend_name}, dataset=${dataset_name}"
                fi
            done
        done
        
        echo "Dataset ${dataset_name} training completed!"
        echo ""
    done
    
    echo "Model backend ${backend_name} all datasets training completed!"
    echo ""
done

echo "========================================"
echo "All models, all datasets training completed!"
echo "========================================"
