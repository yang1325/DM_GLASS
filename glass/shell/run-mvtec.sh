#!/bin/bash

datapath=./datasets/mvtec_ad_dataset
augpath=./datasets/dtd/images

# 定義你要跑的類別
classes=('screw' 'transistor')

# --- 設定實驗名稱的前綴 ---
exp_folder="membank_5shot" 

cd ..

# === [修改 1] 改用迴圈，對每個類別單獨執行 ===
for class in "${classes[@]}"; do
    
    # === [修改 2] 動態設定 run_name ===
    run_name="${class}"
    
    echo "======================================================"
    echo "Running experiment for class: ${class}"
    echo "Run Name: ${run_name}"
    echo "======================================================"

    python main.py \
        --gpu 0 \
        --seed 0 \
        --test ckpt \
        --results_path "results/${exp_folder}" \
        --run_name "${run_name}" \
        --resume \
      net \
        -b wideresnet50 \
        -le layer2 \
        -le layer3 \
        --pretrain_embed_dimension 1536 \
        --target_embed_dimension 1536 \
        --patchsize 3 \
        --meta_epochs 3000 \
        --eval_epochs 10 \
        --dsc_layers 2 \
        --dsc_hidden 1024 \
        --pre_proj 1 \
        --mining 1 \
        --noise 0.015 \
        --radius 0.75 \
        --p 0.5 \
        --step 20 \
        --limit 392 \
      dataset \
        -d "${class}" \
        --distribution 0 \
        --mean 0.5 \
        --std 0.1 \
        --fg 1 \
        --rand_aug 1 \
        --batch_size 8 \
        --resize 288 \
        --imagesize 288 \
        --augment mvtec "$datapath" "$augpath"

done