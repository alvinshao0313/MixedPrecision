import torch
import utils
import model_utils
import transformers
import args_config_gen


def main():
    args = args_config_gen.parser_gen()

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
        # from lm_eval.tasks import TaskManager   # lm_eval==0.4.5
        # from lm_eval import evaluator
        # from lm_eval.utils import simple_parse_args_string
        # from lm_eval.loggers import EvaluationTracker

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    # task_manager = TaskManager()
    # task_names = task_manager.match_tasks(args.tasks)

    results = lm_eval.simple_evaluate(hflm, tasks=task_names,
                                      )['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)

    print("The end")


if __name__ == '__main__':
    main()
