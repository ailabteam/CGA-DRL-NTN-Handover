#!/bin/bash

# run_batch.sh
# Kich hoat moi truong conda neu can (tuong thich voi nhieu loai shell)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate cga_drl

GPU_ID=0
STEPS=200000 
# Danh sach seeds (Chay 5 seeds cho chac chan)
SEEDS=(101 102 103 201 202)

# Danh sach kich ban: Static (Validation) va Random (Generalization - Rotation Noise)
SCENARIOS=("random" "static")

echo "=========================================="
echo "STARTING FINAL BATCH EXPERIMENT (AI + HEURISTIC)"
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
            --seed $seed --steps $STEPS --gpu 0 &
            
        wait

        # 3. Chạy Heuristics (Max-Elevation & Min-Handover)
        # Chay cuc nhanh nen khong can background & wait
        echo ">> [$scenario] Heuristics | Seed: $seed"
        python run_heuristic.py # Script nay tu chay loop seeds/scenarios ben trong? 
        # Khoan, script run_heuristic.py ban viet tu chay loop.
        # Nen o day ta khong can goi no trong vong lap bash.
        # Chi can goi 1 lan sau cung la du.
            
        echo "------------------------------------------"
    done
done

# Chay Heuristics 1 lan duy nhat o cuoi cung (vi no tu loop ben trong python)
echo "##########################################"
echo "### RUNNING HEURISTIC BASELINES ###"
echo "##########################################"
python run_heuristic.py

echo "BATCH EXPERIMENT COMPLETED."
