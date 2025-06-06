if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

GPU_ID=$1
W_BITS=4
A_BITS=16
KV_BITS=16


# 2-7b
CUDA_VISIBLE_DEVICES=$GPU_ID python ./fake_quant/main_for_test.py \
    --model meta-llama/Meta-Llama-3-8B \
    --fuse_norm \
    --use_r1 \
    --use_r2 offline \
    --no-use_r3 \
    --use_r4 \
    --rotate_mode cube_had \
    --rotate_group 16 \
    --w_bits ${W_BITS} \
    --w_rtn \
    --w_groupsize 16 \
    --w_clip \
    --percdamp 0.02 \
    --w_fp4 e2m1_0 \
    --a_bits ${A_BITS} \
    --a_asym \
    --a_groupsize 16 \
    --no-first_token_protect \
    --no-o_per_head \
    --no-mix_precision \
    --protect_threshold 1.8 \
    --a_clip_ratio 1.0 \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --k_asym \
    --v_asym \
    --save_name mp_w${W_BITS}a${A_BITS}kv${KV_BITS} \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 \
    --distribute \
    --lm_eval \
    --lm_eval_batch_size 1 \
    --tasks piqa arc_easy arc_challenge winogrande social_iqa openbookqa \
    # --load_qmodel_path /home/shaoyuantian/program/MixedPrecision/ckpt/quarot/Llama-2-7b-hf_w4_r1_r2_r4_gptq-P0.02_clip_g128
