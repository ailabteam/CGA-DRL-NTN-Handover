#!/bin/bash

# Kích hoạt môi trường conda
# Lưu ý: Cần trỏ đúng đường dẫn conda.sh trên server của bạn nếu chạy từ cron/nohup
# source /opt/anaconda3/etc/profile.d/conda.sh 
# conda activate cga_drl

GPU_ID=0
STEPS=200000 # 200k steps cho mỗi seed (tăng lên 1M cho paper thật)

# Danh sách seeds (Theo checklist Phase 3: nên là 30-50, nhưng test thì dùng 3-5)
SEEDS=(101 102 103 201 202)

echo "=========================================="
echo "STARTING BATCH EXPERIMENT"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    # 1. Chạy Proposed Method (CGA + PPO)
    echo "Running CGA-PPO Seed $seed..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_experiment.py \
        --algo PPO \
        --feature cga \
        --seed $seed \
        --steps $STEPS \
        --gpu 0 &
    
    # Chạy song song process tiếp theo để tối ưu GPU? 
    # KHÔNG NÊN nếu env nặng CPU. Nên chạy tuần tự hoặc dùng GNU Parallel.
    # Ở đây dùng wait để chạy tuần tự cho an toàn.
    wait 

    # 2. Chạy Baseline (XYZ + PPO)
    echo "Running XYZ-PPO (Baseline) Seed $seed..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_experiment.py \
        --algo PPO \
        --feature xyz \
        --seed $seed \
        --steps $STEPS \
        --gpu 0
        
    echo "------------------------------------------"
done

echo "BATCH EXPERIMENT COMPLETED."
