import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import Union
from datasets import load_from_disk
from lm_eval import models


class SVDLinearForWidth(nn.Module):
    def __init__(self, w1, w2, bias=None):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.zeros_like(w1.t()))
        self.w1.data.copy_(w1.t())
        # self.w2 = torch.nn.Parameter(w2.t())

        if bias is not None:
            self.w2 = torch.nn.Linear(w2.shape[0], w2.shape[1], bias=True).to(w2.device)
            self.w2.weight.data.copy_(w2.data.t())
            self.w2.bias.data.copy_(bias)
        else:
            self.w2 = torch.nn.Linear(w2.shape[0], w2.shape[1], bias=False).to(w2.device)
            self.w2.weight.data.copy_(w2.data.t())

    def forward(self, x):
        x = F.linear(x, self.w1)
        out = self.w2(x)

        return out


def get_model(args):
    lm = models.get_model("hf-causal-experimental").create_from_arg_string(
        args.model_args, {"batch_size": 1, "device": 'cuda'}
    )
    return lm


def get_bookcorpus(nsamples, seed, seqlen, tokenizer):

    from datasets import load_dataset, load_from_disk
    traindata = load_from_disk(
        '/root/datasets/bookcorpus/train'
    )
    tokenized_samples, history = [], []
    import random
    random.seed(seed)
    for _ in tqdm(range(nsamples), desc='Sample calibration data from bookcorpus dataset.'):
            while True:
                i = random.randint(0, len(traindata) - 1)
                tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
                if tokenized_sample.input_ids.shape[1] >= seqlen and i not in history:
                    history.append(i)
                    break
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seqlen)
            inp = tokenized_sample.input_ids[:, i:i + seqlen]
            tar = inp.clone()
            tar[:, :-1] = -100
            tokenized_samples.append((inp, tar))

    return tokenized_samples


alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"
}


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name or template_name == 'alpaca':
            self.template = alpaca_template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def get_alpaca(nsamples, seed, seqlen, tokenizer):
    prompter = Prompter('alpaca')
    dataset = load_from_disk('/root/datasets/alpaca/train')
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, return_tensors='pt',
            truncation=True,
            max_length=seqlen
        )

        result["labels"] = result["input_ids"].clone()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt
    import random
    random.seed(seed)
    trainloader, history = [], []
    for _ in tqdm(range(nsamples), desc='Sample calibration data from alpaca dataset.'):
        while True:
            i = random.randint(0, len(dataset) - 1)
            result = generate_and_tokenize_prompt(dataset[i])
            if result['input_ids'].shape[1] >= seqlen and i not in history:
                history.append(i)
                break
        trainloader.append((result['input_ids'], result["labels"]))
    return trainloader


def get_c4(nsamples, seed, seqlen, tokenizer):


    from datasets import load_dataset, load_from_disk
    traindata = load_from_disk('/root/allenai/c4/train')
    # valdata = load_from_disk('allenai/c4/test')

    import random
    random.seed(seed)
    trainloader = []
    for _ in tqdm(range(nsamples)):
        while True:
            #  find a sequence that is longer than seqlen
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        # sample a segment from the above sequence
        try:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        except:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def find_layers(module, layers=[nn.Linear, ], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_out = torch.zeros((self.rows), device=self.dev)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)

        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            out = out.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                out = out.reshape((-1, out.shape[-1]))
            inp = inp.t()
            out = out.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.scaler_out *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        self.scaler_out += torch.norm(out, p=2, dim=1) ** 2 / self.nsamples


def get_config(model_pruned):
    config = model_pruned.config
    mlp_lowranks = []
    q_lowranks = []
    k_lowranks = []
    vo_lowranks = []
    layer_inter_size = []
    for i in range(len(model_pruned.model.layers)):
        layer_pruned = model_pruned.model.layers[i]
        q_lowranks.append(layer_pruned.self_attn.q_proj.w1.shape[0])
        k_lowranks.append(layer_pruned.self_attn.k_proj.w1.shape[0])
        vo_lowranks.append(layer_pruned.self_attn.v_proj.weight.shape[0])
        try:
            layer_inter_size.append(layer_pruned.mlp.gate_proj.weight.shape[0])
            mlp_lowranks.append(0)
        except:
            layer_inter_size.append(layer_pruned.mlp.mlp.gate_proj.weight.shape[0])
            mlp_lowranks.append(layer_pruned.mlp.linear.w1.shape[0])

    config.mlp_lowranks = mlp_lowranks
    config.q_lowranks = q_lowranks
    config.k_lowranks = k_lowranks
    config.vo_lowranks = vo_lowranks
    config.layer_inter_size = layer_inter_size
    model_pruned.config = config
    return config


def solve(inputs, outputs, args):
    H = inputs.t().mm(inputs)
    diag = torch.arange(len(H), device=H.device)
    damp = args.percdamp * torch.mean(torch.diag(H)) / ((args.num_samples / 128) * (args.seqlen / 128))
    H[diag, diag] += damp
    try:
        L = torch.linalg.cholesky(H.cuda()).cpu()
    except:
        L = torch.linalg.cholesky(H).cpu()
    # L_inv = torch.linalg.inv(L)
    # L_inv_T = L_inv.T
    # H_inv = torch.mm(L_inv_T, L_inv).cpu().float()
    H_inv = torch.cholesky_inverse(L).cpu().float()

    solution = (H_inv.mm(inputs.t()).mm(outputs)).cpu()
    pred = inputs.mm(solution)
    importance = 1. - torch.cosine_similarity(outputs, pred).mean()
    L, H_inv, H = L.cpu(), H_inv.cpu(), H.cpu()
    del L, H_inv, H
    torch.cuda.empty_cache()
    return solution, importance, pred
