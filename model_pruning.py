import time
import torch
import argparse
from torch.amp import autocast
from transformers import set_seed
from tqdm import tqdm
import os
import copy
import pickle
from transformers import LlamaForCausalLM as LlamaForCausalLMPretrained
from transformers import LlamaTokenizer
from utils.utils import get_bookcorpus, get_alpaca, get_config
from utils.llama_utils import fast_OND, pruning


@autocast(device_type='cuda')
@torch.no_grad()
def olica_pruning(model, dataloader, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype
    model.eval()
    print("preparing...")
    layer_inputs = torch.zeros((len(dataloader), args.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    inputs_len = []
    for i, data in tqdm(enumerate(dataloader), desc='Prepare input data.'):
        inp = data[0]
        inputs_len.append(args.seqlen)
        try:
            layer_inputs[i] = model.model.embed_tokens(inp)
        except ValueError:
            pass

    attention_mask = torch.ones((1, args.seqlen), dtype=torch.bool, device=layer_inputs.device)
    attention_mask_importance = model.model._prepare_decoder_attention_mask(attention_mask, (1, args.seqlen), layer_inputs, 0)
    layer_inputs_importance = layer_inputs
    position_ids_importance = torch.arange(args.seqlen, dtype=torch.long, device='cpu').unsqueeze(0)

    layer_inputs = copy.deepcopy(layer_inputs_importance)
    attention_mask = copy.deepcopy(attention_mask_importance)
    position_ids = copy.deepcopy(position_ids_importance)

    config = model.config
    model = model.cpu()
    torch.cuda.empty_cache()

    print("pruning...")
    init_inputs = (layer_inputs_importance, attention_mask_importance, position_ids_importance)
    mlp_r_list = fast_OND(model, init_inputs, dtype, config, args)
    mlp_index = mlp_r_list[:-1].argsort()[:args.mlp_num]

    init_inputs = (layer_inputs, attention_mask, position_ids)
    sparsity_qk, sparsity_vp, sparsity_mlp, sparsity = pruning(model, init_inputs, config, mlp_index, args)
    model.config.use_cache = use_cache
    return sparsity_qk, sparsity_vp, sparsity_mlp, sparsity, model

def main(args):
    model_name = args.base_model.split('/')[-1]
    print('load model...')
    model = LlamaForCausalLMPretrained.from_pretrained(args.base_model, torch_dtype='auto', device_map='cpu')
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model.eval()

    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k, v in state_dict.items() if k not in ('model.embed_tokens.weight', 'lm_head.weight')) / 10 ** 9,2)
    extra_params = round(sum(v.numel() for k, v in state_dict.items() if k in ('model.embed_tokens.weight', 'lm_head.weight')) / 10 ** 9,2)

    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')
    print('load dataset...')

    data_path = './data/{}_{}_nsample:{}_seqlen{}.pkl'.format(model_name, args.datasets, args.num_samples, args.seqlen)
    dataloader = []
    if not os.path.exists(data_path):
        if 'bookcorpus' in args.datasets:
            dataloader += get_bookcorpus(nsamples=args.num_samples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
        if 'alpaca' in args.datasets:
            dataloader += get_alpaca(nsamples=args.num_samples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
        with open(data_path, 'wb') as f:
            pickle.dump(dataloader, f)
    else:
        with open(data_path, 'rb') as f:
            dataloader = pickle.load(f)

    num_samples = len(dataloader)

    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')

    print('start olica pruning...')
    if max(args.sparsity, args.mlp_num) > 0:
        tick = time.time()
        sparsity_qk, sparsity_vp, sparsity_mlp, sparsity, model = olica_pruning(model, dataloader, args)
        total_time = time.time() - tick
        s = 'total_time:{:.2f}, sparsity_qk:{:.2f}, sparsity_vp:{:.2f}, sparsity_mlp:{:.2f}, Total sparsity:{:.2f}'.format(total_time, sparsity_qk, sparsity_vp, sparsity_mlp, sparsity)
        print(s)

    if args.save_dir:
        from utils.customized_llama import LlamaForCausalLM
        model.half()
        model.cpu()
        config = get_config(model)
        pruned_model = LlamaForCausalLM(config)
        pruned_model.half()
        pruned_model.load_state_dict(model.state_dict(), strict=True)
        save_dir = os.path.join(args.save_dir, model_name, 'SR:{}_{}'.format(args.sparsity, model_name))
        print('save model...')
        pruned_model.save_pretrained(save_dir)
        print('save tokenizer...')
        tokenizer.save_pretrained(save_dir)
        print('save dir:', save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='7b', help="model to load")
    parser.add_argument("--datasets", type=str,default='bookcorpus+alpaca',help="Where to extract calibration data from.",)
    parser.add_argument("--num_samples", type=int, default=128,help="Number of calibration data samples.")
    parser.add_argument("--seqlen", type=int, default=128,help="Sequence length for the calibration data.")
    parser.add_argument("--sparsity", type=float, default=0.0,help="Sparsity ratio")
    parser.add_argument("--cache_dev", type=str, default="cuda",help="Defaults to `cuda`. When the GPU memory is insufficient, you can set `cache_dev` to `cpu`, but the trade-off is slower pruning speed.")
    parser.add_argument("--save_dir", type=str, default="",help="Path to saved model.",)

    parser.add_argument("--percdamp", type=float, default=0.5,help="Percent of the average Hessian diagonal to use for dampening.",)
    parser.add_argument("--ratio", type=float, default=0.03,help="Rank ratio for the SVD of linear calibration",)
    parser.add_argument("--seed", type=int, default=0,help="Seed for sampling the calibration data.")
    parser.add_argument("--mlp_num", type=int, default=0,help="Number of MLP layers to calibrate.")
    parser.add_argument("--k", type=float, default=3.,help="ratio of QK vs VO.")
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    model_name = 'Llama-{}'.format(args.base_model)
    args.base_model = '/userhome/home/hejiujun/ckpts/Llama-{}'.format(args.base_model)
    main(args)