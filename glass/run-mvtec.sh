#!/bin/bash

datapath=./datasets/mvtec_ad_dataset
augpath=./datasets/dtd/images

# --- 要跑的類別 ---
classes=('transistor' 'screw')

# --- 設定實驗名稱 ---
exp_folder="CLR_membank_x3_5shot" 

cd ..

for class in "${classes[@]}"; do
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
        --meta_epochs 2000 \
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
        --memory_bank_type sampled \
        --sampling_ratio 0.1 \
      dataset \
        -d "${class}" \
        --distribution 0 \
        --mean 0.5 \
        --std 0.1 \
        --fg 1 \
        --rand_aug 1 \
        --batch_size 32 \
        --resize 288 \
        --imagesize 288 \
        --k_shot 5 \
        --augment mvtec "$datapath" "$augpath"

done