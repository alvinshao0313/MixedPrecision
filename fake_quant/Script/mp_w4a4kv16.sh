GPU_ID=0
W_BITS=4
A_BITS=4
KV_BITS=16


# 2-7b
CUDA_VISIBLE_DEVICES=$GPU_ID python ./fake_quant/main_for_test.py \
    --model meta-llama/Llama-2-7b-hf \
    --fuse_norm \
    --use_r1 \
    --use_r2 offline \
    --no-use_r3 \
    --use_r4 \
    --no-w_rtn \
    --w_groupsize -1 \
    --w_clip \
    --mix_precision \
    --protect_threshold 1.8 \
    --a_clip_ratio 1.0 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --no-first_token_protect \
    --k_asym \
    --v_asym \
    --o_per_head \
    --percdamp 0.1 \
    --save_name mp_w${W_BITS}a${A_BITS}kv${KV_BITS} \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 \
    --distribute \
    --lm_eval \
    --lm_eval_batch_size 16 \
    --tasks piqa arc_easy arc_challenge winogrande social_iqa openbookqa \
    --load_qmodel_path /home/shaoyuantian/program/MixedPrecision/ckpt/quarot/Llama-2-7b-hf_w4_r1_r2_r4_gptq-P0.02_clip
    # --load_qmodel_path /home/shaoyuantian/program/MixedPrecision/ckpt/quarot/Llama-2-7b-hf_w4_r1_r2_r4_gptq_clip \

# # 2-13b
# CUDA_VISIBLE_DEVICES=$GPU_ID python main_for_test.py \
#     --model meta-llama/Llama-2-13b-hf \
#     --fuse_norm \
#     --use_r1 \
#     --use_r2 online \
#     --no-use_r3 \
#     --use_r4 \
#     --w_groupsize -1 \
#     --w_clip \
#     --a_clip_ratio 0.9 \
#     --w_bits ${W_BITS} \
#     --a_bits ${A_BITS} \
#     --k_bits ${KV_BITS} \
#     --v_bits ${KV_BITS} \
#     --k_groupsize 128 \
#     --v_groupsize 128 \
#     --k_asym \
#     --v_asym \
#     --o_per_head \
#     --percdamp 0.1 \
#     --no-w_ft \
#     --ft_percdamp 0.0 \
#     --save_name dart_w${W_BITS}a${A_BITS}kv${KV_BITS} \
#     --save_qmodel_path /home/shaoyuantian/program/RLLM/fake_quant/ckpt/wikitext2_128samples/ \
#     --distribute \
#     --lm_eval \
#     --lm_eval_batch_size 2 \
#     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu \
#     --ppl_eval \
#     --ppl_eval_batch_size 1 \
#     --ppl_eval_dataset wikitext2 ptb c4 \

# # 3-8b
# CUDA_VISIBLE_DEVICES=$GPU_ID python main_for_test.py \
#     --model meta-llama/Meta-Llama-3-8B \
#     --fuse_norm \
#     --use_r1 \
#     --use_r2 online \
#     --no-use_r3 \
#     --use_r4 \
#     --w_groupsize -1 \
#     --w_clip \
#     --a_clip_ratio 0.9 \
#     --w_bits ${W_BITS} \
#     --a_bits ${A_BITS} \
#     --k_bits ${KV_BITS} \
#     --v_bits ${KV_BITS} \
#     --k_groupsize 128 \
#     --v_groupsize 128 \
#     --k_asym \
#     --v_asym \
#     --o_per_head \
#     --percdamp 0.1 \
#     --no-w_ft \
#     --ft_percdamp 0.0 \
#     --save_name dart_w${W_BITS}a${A_BITS}kv${KV_BITS} \
#     --save_qmodel_path /home/shaoyuantian/program/RLLM/fake_quant/ckpt/wikitext2_128samples/ \
#     --distribute \
#     --lm_eval \
#     --lm_eval_batch_size 16 \
#     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu \
#     --ppl_eval \
#     --ppl_eval_batch_size 1 \
#     --ppl_eval_dataset wikitext2 ptb c4 \