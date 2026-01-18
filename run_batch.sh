#!/bin/bash

# run_batch.sh
# Kich hoat moi truong conda neu can
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate cga_drl

GPU_ID=0
STEPS=200000 
# Danh sach seeds (Chay 3-5 seeds truoc de test)
SEEDS=(101 102 103 201 202)

# Danh sach kich ban: Static (Validation) va Random (Generalization)
SCENARIOS=("random" "static")

echo "=========================================="
echo "STARTING FULL BATCH EXPERIMENT"
echo "=========================================="

for scenario in "${SCENARIOS[@]}"; do
    echo "##########################################"
    echo "### RUNNING SCENARIO: $scenario ###"
    echo "##########################################"

    for seed in "${SEEDS[@]}"; do
        # 1. Chạy CGA-PPO
        echo ">> [$scenario] CGA-PPO | Seed: $seed"
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_experiment.py \
            --algo PPO --feature cga --scenario $scenario \
            --seed $seed --steps $STEPS --gpu 0 &
        
        wait 

        # 2. Chạy XYZ-PPO (Baseline)
        echo ">> [$scenario] XYZ-PPO | Seed: $seed"
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_experiment.py \
            --algo PPO --feature xyz --scenario $scenario \
            --seed $seed --steps $STEPS --gpu 0
            
        echo "------------------------------------------"
    done
done

echo "BATCH EXPERIMENT COMPLETED."
