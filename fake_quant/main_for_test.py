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
from functools import partial


def main():
    args = args_config_gen.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model.model_name = args.model.split('/')[-1]

    # Rotate the weights
    if args.fuse_norm:
        logging.info("Fuse LayerNorms")
        logging.info("Rotate the model use_r1={}, use_r2={}, use_r4={}, use_r3={}".format(
            args.use_r1, args.use_r2, args.use_r4, args.use_r3))

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
    elif args.w_bits < 16 or args.a_bits < 16 or args.v_bits < 16 or args.k_bits < 16:
        # Add Activation Wrapper to the model as the rest of the code assumes it is present
        quant_utils.add_actquant(model)

    if args.w_bits < 16:
        logging.info("Add weight quantization: w_rtn = {}, w_bits = {}, w_groupsize = {}, w_sym = {}, w_clip = {}".format(
            args.w_rtn, args.w_bits, args.w_groupsize, not (args.w_asym), args.w_clip))

        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            # assert args.fuse_norm, "Model should be fused to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            logging.info("Load quantized model from: {}.".format(args.load_qmodel_path))
            utils.load_model_in_parts(model, args.load_qmodel_path)
            # save_dict = torch.load(args.load_qmodel_path, map_location='cpu')
            # model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"

            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            folder_name = f'{model.model_name}'
            folder_name += f'_w{args.w_bits}'
            folder_name += '_r1' if args.use_r1 else ''
            folder_name += '_r2' if args.use_r2 != 'none' else ''
            folder_name += '_r3' if args.use_r3 else ''
            folder_name += '_r4' if args.use_r4 else ''
            folder_name += '_rtn' if args.w_rtn else f'_gptq-P{args.percdamp}'
            folder_name += '_clip' if args.w_clip else ''
            folder_name += f'_g{args.w_groupsize}' if args.w_groupsize > 0 else ''
            folder_name += '_asym' if args.w_asym else ''
            folder_name += '_smooth' if args.smooth else ''
            folder_name += f'_W{args.w_fp4}' if args.w_fp4 else ''
            args.save_qmodel_path = os.path.join(args.save_qmodel_path, folder_name)
            if not os.path.exists(args.save_qmodel_path):
                os.makedirs(args.save_qmodel_path)
            logging.info("Save quantized model to: {}.".format(args.save_qmodel_path))
            utils.save_model_in_parts(model, args.save_qmodel_path, prefix=f'{model.model_name}_part')
            # save_dict["model"] = model.state_dict()
            # torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        logging.info("Add v quantization: v_bits={}, v_groupsize={}, v_sym={}, v_clip_ratio={}".format(
            args.v_bits, args.v_groupsize, not (args.v_asym), args.v_clip_ratio))

        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        # if args.up_out_bits < 16:
        #     had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        #     fixed_quant_act_fn = partial(quant_utils.quant_act_fn,
        #                                  act_fn=model.config.hidden_act,
        #                                  q_config={
        #                                      'bits': args.up_out_bits,
        #                                      'groupsize': args.up_groupsize,
        #                                      'sym': not (args.a_asym),
        #                                      'clip_ratio': args.a_clip_ratio,
        #                                      "first_token_protect": args.first_token_protect,
        #                                      "residual": args.a_residual,
        #                                      "mixed_precision": args.mixed_precision,
        #                                      "protect_threshold": args.protect_threshold
        #                                  })
        #     for name, moudle in model.named_modules():
        #         if hasattr(module, "act_fn"):
        #             moudle.act_fn = quant_utils.quant_act_fn()

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            residual = args.a_residual
            first_token_protect = args.first_token_protect
            mix_precision = args.mix_precision
            protect_threshold = args.protect_threshold

            # if args.up_out_bits < 16 and 'up_proj' in name:
            #     qlayers[name].out_quantizer.configure(bits=args.up_out_bits,
            #                                           groupsize=args.up_groupsize,
            #                                           sym=not (args.a_asym),
            #                                           clip_ratio=args.a_clip_ratio,
            #                                           first_token_protect=first_token_protect,
            #                                           residual=residual,
            #                                           mix_precision=mix_precision,
            #                                           protect_threshold=protect_threshold)

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
                                              residual=residual,
                                              first_token_protect=first_token_protect,
                                              mix_precision=args.mix_precision,
                                              protect_threshold=protect_threshold,
                                              fp4=args.a_fp4,)

    if args.k_bits < 16:
        logging.info("Add k quantization: k_bits={}, k_groupsize={}, k_sym={}, k_clip_ratio={}".format(
            args.k_bits, args.k_groupsize, not (args.k_asym), args.k_clip_ratio))

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

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    if args.ppl_eval:
        logging.info("Evaluating PPL on datasets: {}".format(args.ppl_eval_dataset))
        for dataset in args.ppl_eval_dataset:
            testenc = data_utils.get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                eval_mode=True)
            dataset_ppl, total_tokens = eval_utils.ppl_evaluator(model, testenc, utils.DEV, args)
            protect_ratio = quant_utils.protection_ratio(model, total_tokens)
            logging.info(f'{dataset.upper()} PPL: {dataset_ppl:.2f}, {protect_ratio:.2f} % protected')

            if not args.log_to_console:
                print(f'{dataset.upper()} PPL: {dataset_ppl:.2f}')

            if args.wandb:
                wandb.log({'ppl/{}'.format(dataset.upper()): dataset_ppl})

    if args.lm_eval:
        logging.info("Evaluating on downstream tasks: {}".format(args.tasks))
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager   # lm_eval==0.4.3

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)  # args.lm_eval_batch_size "auto"

        task_manager = TaskManager()
        task_names = task_manager.match_tasks(args.tasks)

        results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=32)['results']

        metric_vals = {task: round(result.get(
            'acc_norm,none', result['acc,none']) * 100, 2) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)

        logging.info(metric_vals)

        if args.wandb:
            wandb.log(metric_vals)

    logging.info('--' * 30 + '\n\n')
    print("The end")


if __name__ == '__main__':
    main()
