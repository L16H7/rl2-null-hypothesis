#!/bin/bash

# Script to run experiments with 5 different random seeds
# Runs all 4 configurations sequentially for each seed

set -e  # Exit on any error

# Configuration
SEEDS=(42 123 456 789 1337)
CONFIG_FILES=(
    "configs/h0.yaml"
    "configs/h1.yaml"
    "configs/prev_action.yaml"
    "configs/prev_reward.yaml"
)
CONFIG_NAMES=("H0" "H1" "PREV_ACTION" "PREV_REWARD")
SCRIPT_DIR="train.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RL2 Null Hypothesis Experiment${NC}"
echo -e "${BLUE}  Running 4 configs with 5 seeds${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run experiment
run_experiment() {
    local config_file=$1
    local seed=$2
    local exp_type=$3
    local save_dir="experiment_${exp_type}_${seed}"
    
    echo -e "${YELLOW}Starting $exp_type experiment with seed $seed...${NC}"
    echo -e "${YELLOW}Config: $config_file${NC}"
    echo -e "${YELLOW}Save directory: $save_dir${NC}"
    echo -e "${YELLOW}Command: python $SCRIPT_DIR --config_path $config_file --train_seed $seed --eval_seed $seed --checkpoint_path $save_dir${NC}"
    echo ""
    
    # Create save directory
    mkdir -p "$save_dir"
    
    # Run the experiment
    if python $SCRIPT_DIR --config_path $config_file --train_seed $seed --eval_seed $seed --checkpoint_path "$save_dir"; then
        echo -e "${GREEN}✓ $exp_type experiment with seed $seed completed successfully${NC}"
    else
        echo -e "${RED}✗ $exp_type experiment with seed $seed failed${NC}"
        return 1
    fi
    echo ""
}

# Main experiment loop
total_experiments=$((${#SEEDS[@]} * ${#CONFIG_FILES[@]}))
current_experiment=0

for seed in "${SEEDS[@]}"; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Seed: $seed${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Run all configurations for this seed
    for i in "${!CONFIG_FILES[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo -e "${BLUE}Experiment $current_experiment/$total_experiments${NC}"
        run_experiment "${CONFIG_FILES[$i]}" "$seed" "${CONFIG_NAMES[$i]}"
    done
    
    echo -e "${GREEN}Completed all configurations for seed $seed${NC}"
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All experiments completed!${NC}"
echo -e "${GREEN}  Total experiments run: $total_experiments${NC}"
echo -e "${GREEN}========================================${NC}"
