import torch
import argparse
from transformers import set_seed
from lm_eval import tasks, utils
from evaluate import evaluate
from ppl_eval.ppl_eval import ppl_metric
from utils.customized_llama import LlamaForCausalLM, AutoTokenizer
from utils.utils import get_model


def main(args):
    model_name = args.base_model.split('/')[-1]
    args.log_file = open('{}_log.txt'.format(model_name), mode='a')
    print('load model...')
    lm = get_model(args)
    lm.model = lm.model.cpu()
    torch.cuda.empty_cache()
    del lm.model

    path = args.save_dir + '/{}/SR:{}_{}'.format(model_name, args.sparsity, model_name)

    model = LlamaForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16)
    model.eval()
    lm.model = model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    args.model = lm
    print('start evaluate...')
    metric = ppl_metric(model, tokenizer, ['wikitext2'], 128, 2)
    mean, table = evaluate(lm, tokenizer, args)
    args.log_file.write('SR: {}, PPL: {}, Mean: {} \n {}\n'.format(args.sparsity, metric['wikitext2'], mean, table))
    args.log_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='7b', help="model to load")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Target pruning ratio")
    parser.add_argument("--save_dir", type=str, default="",help="Path to saved model.",)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.",)

    # eval args
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=32)
    parser.add_argument("--max_batch_size", type=int, default=None, help="Maximal batch size to try with --batch_size auto",)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit",type=float,default=None,
        help="Limit the number of examples per task. "
             "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--model_args", type=str, default='')
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    model_name = 'Llama-{}'.format(args.base_model)
    args.model_args = "pretrained=/userhome/home/hejiujun/ckpts/Llama-{}".format(args.base_model)
    args.base_model = '/userhome/home/hejiujun/ckpts/Llama-{}'.format(args.base_model)
    main(args)

