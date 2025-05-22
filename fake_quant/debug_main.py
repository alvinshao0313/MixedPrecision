import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import args_config_gen
import hadamard_utils
import logging
import os
import w_fine_tuning


def main():
    args = args_config_gen.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # Rotate the weights
    if args.use_r1 or args.use_r2 != 'none' or args.use_r4:
        rotation_utils.fuse_layer_norms(model)
        if args.use_r1 or args.use_r2 != 'none' or args.use_r4:
            rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if args.use_r4 and 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if args.use_r2 == 'online' and 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        # Add Activation Wrapper to the model as the rest of the code assumes it is present
        quant_utils.add_actquant(model)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            # assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"

            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            # 精度补偿：
            if args.w_ft:
                w_fine_tuning.w_ft(model, trainloader, utils.DEV, args)
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization

            if args.w_ft:  # 精度补偿：
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                w_fine_tuning.w_ft(model, trainloader, utils.DEV, args)
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            residual = args.a_residual
            static = args.a_static

            if 'v_proj' in name and args.v_bits < 16:  # Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                      groupsize=args.v_groupsize,
                                                      sym=not (args.v_asym),
                                                      clip_ratio=args.v_clip_ratio)

            if 'lm_head' in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if args.o_per_head and 'o_proj' in name:  # Set the o_proj precision
                num_heads = model.config.num_attention_heads
                model_dim = model.config.hidden_size
                layer_groupsize = model_dim // num_heads

            if 'down_proj' in name:  # Set the down_proj precision
                if args.a_bits_down_proj is not None:
                    layer_input_bits = args.a_bits_down_proj
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              residual=residual)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits': args.k_bits, "k_groupsize": args.k_groupsize,
                              "k_sym": not (args.k_asym), "k_clip_ratio": args.k_clip_ratio,
                              'use_r3': args.use_r3}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config)

    # Evaluating on dataset
    cache_testloader = f"{args.save_path.rsplit('/', 1)[0]}/{args.model.rsplit('/', 1)[-1]}_testloader_wikitext2.cache"
    print(cache_testloader)
    if os.path.exists(cache_testloader):
        testloader = torch.load(cache_testloader)
        logging.info(f"load calibration from {cache_testloader}")
    else:
        testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
        torch.save(testloader, cache_testloader)

    dataset_ppl = eval_utils.ppl_evaluator(model, testloader, utils.DEV, args)
    if not args.log_to_console:
        print(f'{args.eval_dataset.upper()} PPL: {dataset_ppl:.3f}')

    if args.wandb:
        wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)

    if args.wandb:
        wandb.log(metric_vals)

    logging.info('\n\n' + '--' * 30 + '\n\n')
    print("The end")


if __name__ == '__main__':
    main()
