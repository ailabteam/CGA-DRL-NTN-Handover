#!/bin/bash
# run_test_batch.sh

# Chạy mỗi cái 20k bước, 2 seeds
STEPS=20000 
SEEDS=(101 102) 
SCENARIOS=("random") # Test kịch bản khó nhất

echo "=== STARTING QUICK TEST BATCH ==="

# Xóa log cũ để test cho sạch
rm -rf results/logs/random/*

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">> Testing CGA-PPO | Seed: $seed"
        # Bỏ CUDA_VISIBLE_DEVICES vì chạy CPU
        python train_experiment.py --algo PPO --feature cga --scenario $scenario --seed $seed --steps $STEPS &
        
        echo ">> Testing XYZ-PPO | Seed: $seed"
        python train_experiment.py --algo PPO --feature xyz --scenario $scenario --seed $seed --steps $STEPS &
        
        wait
    done
done

# Chạy Heuristic 1 lần (nó nhanh sẵn rồi)
python run_heuristic.py

echo "=== TEST DONE. GENERATING REPORT ==="
python src/utils/paper_reporter.py
