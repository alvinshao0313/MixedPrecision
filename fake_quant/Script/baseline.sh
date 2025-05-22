#!/bin/bash

# 检查是否提供了1个参数：GPU编号
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL>"
    exit 1
fi

# 获取传入的参数
GPU_ID=$1
MODEL=$2

# 提取“/”后面的字符串
MODEL_NAME=${MODEL##*/}

# 执行命令并使用传入的参数
CUDA_VISIBLE_DEVICES=${GPU_ID} python main_for_test.py \
    --model ${MODEL} \
    --w_bits 16 \
    --a_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --no-fuse_norm \
    --no-use_r1 \
    --use_r2 none \
    --no-use_r3 \
    --no-use_r4 \
    --w_groupsize -1 \
    --a_clip_ratio 1.0 \
    --k_groupsize -1 \
    --v_groupsize -1 \
    --percdamp 0.1 \
    --no-w_ft \
    --ft_percdamp 0.0 \
    --save_name baseline-fp16 \
    --distribute \
    --lm_eval \
    --lm_eval_batch_size 4 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 ptb c4 \
