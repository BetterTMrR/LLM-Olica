from tqdm import tqdm
import math
import torch
from torch.nn.parameter import Parameter
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from torch.amp import autocast
from typing import Optional, Tuple
import torch.nn as nn
from utils.utils import find_layers, WrappedGPT, solve, SVDLinearForWidth
import numpy as np


@autocast(device_type='cuda')
def thinner_mlp(mlp, wrapped_layers, hidden_dim, ratio_mlp=0.25):
    W_metric_up = torch.abs(mlp.up_proj.weight.data) * torch.sqrt(wrapped_layers['mlp.up_proj'].scaler_row.reshape((1, -1)))
    W_metric_gate = torch.abs(mlp.gate_proj.weight.data) * torch.sqrt(wrapped_layers['mlp.gate_proj'].scaler_row.reshape((1, -1)))
    W_metric_down = torch.abs(mlp.down_proj.weight.data) * torch.sqrt(wrapped_layers['mlp.down_proj'].scaler_row.reshape((1, -1)))
    neuron_impt = torch.norm(W_metric_up, dim=1, p=2) + torch.norm(W_metric_gate, dim=1, p=2) + torch.norm(W_metric_down, dim=0, p=2)

    v, index = neuron_impt.sort()
    dna = int(0.01 * len(neuron_impt))

    index_pruned = index[dna: dna + int(ratio_mlp * len(neuron_impt))]
    index_kept = torch.cat([index[:dna], index[dna + int(ratio_mlp * len(neuron_impt)):]])

    assert (len(index_pruned) + len(index_kept)) == len(v)

    mlp.up_proj.in_features = hidden_dim
    mlp.up_proj.ou_features = len(index_kept)

    mlp.gate_proj.in_features = hidden_dim
    mlp.gate_proj.ou_features = len(index_kept)

    mlp.down_proj.in_features = len(index_kept)
    mlp.down_proj.ou_features = hidden_dim

    mlp.up_proj.weight = Parameter(mlp.up_proj.weight.data[index_kept])
    mlp.gate_proj.weight = Parameter(mlp.gate_proj.weight.data[index_kept])
    mlp.down_proj.weight = Parameter(mlp.down_proj.weight.data[:, index_kept])

    return mlp


class CustomizedMLP(nn.Module):
    def __init__(self, mlp, linear):
        super().__init__()
        self.mlp = mlp
        self.linear = linear

    def forward(self, hidden_states):
        out1 = self.mlp(hidden_states)
        out2 = self.linear(hidden_states)
        return out1 + out2


def fast_OND(model, init_inputs, dtype, config, args):
    layer_inputs, attention_mask, position_ids = init_inputs
    feat_dim = config.hidden_size
    mlp_r_list = []
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc='Fast Orthogonal Neuron Decomposition'):

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = 'cuda'

        layer = layers[i].to(dev)
        subset = find_layers(layer)
        wrapped_layers = {}

        decoder_layer_inps = []
        attn_oups = []
        attn_inps = []
        mlp_oups = []
        mlp_inps = []

        def tmp_decoder_layer_inps(_, inp, out):
            decoder_layer_inps.append(inp[0].data.float().reshape(-1, feat_dim).cpu())

        def tmp_attn_inps(_, inp, out):
            attn_inps.append(inp[0].data.float().reshape(-1, feat_dim).cpu())

        def tmp_attn_oups(_, inp, out):
            attn_oups.append(out[0].data.float().reshape(-1, feat_dim).cpu())

        def tmp_mlp_inps_oups(_, inp, out):
            mlp_oups.append(out[0].data.float().reshape(-1, feat_dim).cpu())
            mlp_inps.append(inp[0].data.float().reshape(-1, feat_dim).cpu())

        h1 = layer.register_forward_hook(tmp_decoder_layer_inps)
        h2 = layer.self_attn.q_proj.register_forward_hook(tmp_attn_inps)
        h3 = layer.self_attn.register_forward_hook(tmp_attn_oups)
        h4 = layer.mlp.register_forward_hook(tmp_mlp_inps_oups)

        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(len(layer_inputs)):
            with torch.no_grad():
                layer_inputs[j] = \
                    layer(layer_inputs[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0].to(layer_inputs).cpu()

        for h in handles + [h1, h2, h3, h4]:
            h.remove()

        decoder_layer_inps = torch.cat(decoder_layer_inps)

        attn_inps = torch.cat(attn_inps)
        attn_oups = torch.cat(attn_oups)

        mlp_inps = torch.cat(mlp_inps)
        mlp_oups = torch.cat(mlp_oups)

        solution_mlp, _, pred_mlp = solve(mlp_inps, mlp_oups, args)
        r_mlp = ((pred_mlp - pred_mlp.mean(0, keepdim=True)) * (mlp_oups - mlp_oups.mean(0, keepdim=True))).mean(
            0) / (pred_mlp.std(0) * mlp_oups.std(0))
        mlp_r_list.append(r_mlp)

        WD_v = subset['self_attn.v_proj'].weight.data.reshape(layer.self_attn.num_heads, -1, feat_dim)
        WD_o = subset['self_attn.o_proj'].weight.data * torch.sqrt(wrapped_layers['self_attn.o_proj'].scaler_out.reshape((-1, 1)))
        WD_o = WD_o.t().reshape(layer.self_attn.num_heads, -1, feat_dim)
        U, S, V = torch.linalg.svd(WD_o.float().to(dev), full_matrices=False)
        R = U
        L = V * S.unsqueeze(-1) / torch.sqrt(wrapped_layers['self_attn.o_proj'].scaler_out.reshape((1, 1, -1)))
        V_hat = torch.matmul(WD_v.float().transpose(1, 2), R).transpose(1, 2).reshape(-1, feat_dim)
        O_hat = L.reshape(-1, feat_dim).t()
        subset['self_attn.v_proj'].weight.data = V_hat.to(dtype)
        subset['self_attn.o_proj'].weight.data = O_hat.to(dtype)

        torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        U, S, V, L = U.cpu(), S.cpu(), V.cpu(), L.cpu()
        del layer, U, S, R, L, V
    torch.cuda.empty_cache()
    mlp_r_list = 1. - torch.stack(mlp_r_list, dim=0).cpu().detach().mean(1)
    return mlp_r_list


def sr_allocation(layers, n_pruned_params, config, args):
    if args.sparsity <= 0.4:
        sparsity_attn = args.sparsity
        qk_vp_ratio = args.k

        n_attn_params_total = sum([p.numel() for p in layers[0].self_attn.parameters()])
        n_mlp_params_total = sum([p.numel() for p in layers[0].mlp.parameters()])

        sparsity_vp = 2 * sparsity_attn / (1 + qk_vp_ratio)
        sparsity_qk = args.sparsity * 2

        n_pruned_params -= (config.num_hidden_layers * n_attn_params_total / 2 * sparsity_vp)
        n_mlp_params = (n_pruned_params - (n_attn_params_total * args.sparsity) * config.num_hidden_layers) / config.num_hidden_layers
        sparsity_mlp = n_mlp_params / n_mlp_params_total

    else:
        sparsity_attn = args.sparsity
        qk_vp_ratio = args.k

        n_attn_params_total = sum([p.numel() for p in layers[0].self_attn.parameters()])
        n_mlp_params_total = sum([p.numel() for p in layers[0].mlp.parameters()])

        sparsity_vp_t = 2 * sparsity_attn / (1 + qk_vp_ratio)
        sparsity_qk = qk_vp_ratio * sparsity_vp_t
        sparsity_vp = sparsity_vp_t

        n_mlp_params = (n_pruned_params - n_attn_params_total * args.sparsity * config.num_hidden_layers) / config.num_hidden_layers
        sparsity_mlp = n_mlp_params / n_mlp_params_total
    return sparsity_qk, sparsity_vp, sparsity_mlp


def pruning(model, init_inputs, config, mlp_index, args):
    layer_inputs, attention_mask, position_ids = init_inputs

    layers = model.model.layers
    feat_dim = config.hidden_size

    n_total_params = sum([p.numel() for p in model.parameters()])
    n_pruned_params = n_total_params * args.sparsity

    sparsity_qk, sparsity_vp, sparsity_mlp = sr_allocation(layers, n_pruned_params, config, args)

    for layer_index in tqdm(range(len(layers)), desc='Pruning Width'):
        if f"model.layers.{layer_index}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{layer_index}"]
        else:
            dev = 'cuda'

        layer = layers[layer_index].to(dev)
        subset = find_layers(layer)
        attn_oups = []
        attn_inps = []

        mlp_oups = []
        mlp_inps = []

        def tmp_attn_inps(_, inp, out):
            attn_inps.append(inp[0].data.float().reshape(-1, feat_dim).cpu())

        def tmp_attn_oups(_, inp, out):
            attn_oups.append(out[0].data.float().reshape(-1, feat_dim).cpu())

        def tmp_mlp_inps_oups(_, inp, out):
            mlp_oups.append(out[0].data.float().reshape(-1, feat_dim).cpu())
            mlp_inps.append(inp[0].data.float().reshape(-1, feat_dim).cpu())

        h2 = layer.self_attn.q_proj.register_forward_hook(tmp_attn_inps)
        h3 = layer.self_attn.register_forward_hook(tmp_attn_oups)
        h4 = layer.mlp.register_forward_hook(tmp_mlp_inps_oups)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []

        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(len(layer_inputs)):
            with torch.no_grad():
                layer(layer_inputs[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0].to(layer_inputs).cpu()

        for h in handles:
            h.remove()

        h2.remove()
        h3.remove()
        h4.remove()

        attn_inps = torch.cat(attn_inps)
        attn_oups = torch.cat(attn_oups)

        mlp_inps = torch.cat(mlp_inps)
        mlp_oups = torch.cat(mlp_oups)

        # Query and Key
        WD_q = subset['self_attn.q_proj'].weight.data * torch.sqrt(
            wrapped_layers['self_attn.q_proj'].scaler_row.reshape((1, -1)))
        U, S, V = torch.linalg.svd(WD_q.to(dev))

        r = int(np.ceil((1. - sparsity_qk) * feat_dim / 2))
        R = V / torch.sqrt(wrapped_layers['self_attn.q_proj'].scaler_row.reshape((1, -1)))
        layer.self_attn.q_proj = SVDLinearForWidth(R[:r].t(), (U[:, :r] * S[:r].reshape(1, -1)).t())

        WD_k = subset['self_attn.k_proj'].weight.data * torch.sqrt(
            wrapped_layers['self_attn.k_proj'].scaler_row.reshape((1, -1)))
        U, S, V = torch.linalg.svd(WD_k.to(dev))
        r = int(np.ceil((1. - sparsity_qk) * subset['self_attn.k_proj'].weight.data.shape[0] / 2))
        R = V / torch.sqrt(wrapped_layers['self_attn.k_proj'].scaler_row.reshape((1, -1)))
        layer.self_attn.k_proj = SVDLinearForWidth(R[:r].t(), (U[:, :r] * S[:r].reshape(1, -1)).t())

        # Value and O_proj

        W_metric_v = torch.abs(subset['self_attn.v_proj'].weight.data) * torch.sqrt(
            wrapped_layers['self_attn.v_proj'].scaler_row.reshape((1, -1)))
        W_metric_o = torch.abs(subset['self_attn.o_proj'].weight.data) * torch.sqrt(
            wrapped_layers['self_attn.o_proj'].scaler_row.reshape((1, -1)))
        neuron_impt_vp = torch.norm(W_metric_v, dim=1, p=2) + torch.norm(W_metric_o, dim=0, p=2)
        neuron_impt_vp = neuron_impt_vp.reshape(layer.self_attn.num_heads, -1)
        v, indexes = neuron_impt_vp.sort(dim=1)

        indexes = indexes[:, int(sparsity_vp * v.shape[1]):]
        indexes = indexes + v.shape[1] * torch.arange(v.shape[0], device=v.device).reshape(v.shape[0], 1)
        indexes = indexes.reshape(-1)

        inter_dim = feat_dim - int(sparsity_vp * v.shape[1]) * v.shape[0]

        layer.self_attn.v_proj.in_features = feat_dim
        layer.self_attn.v_proj.out_features = inter_dim

        layer.self_attn.o_proj.in_features = inter_dim
        layer.self_attn.o_proj.out_features = feat_dim

        layer.self_attn.v_proj.weight = Parameter(subset['self_attn.v_proj'].weight.data[indexes])
        layer.self_attn.o_proj.weight = Parameter(subset['self_attn.o_proj'].weight.data[:, indexes])

        layer.self_attn.forward = forward.__get__(layer.self_attn, torch.nn.Module)

        mlp = thinner_mlp(layer.mlp, wrapped_layers, feat_dim, ratio_mlp=sparsity_mlp)

        if layer_index in mlp_index:
            thinner_mlp_oups_list = []
            assert  args.seqlen * args.num_samples == len(mlp_inps)
            for i in range(args.num_samples):
                thinner_mlp_oups = mlp(mlp_inps[i*args.seqlen:(i+1)*args.seqlen].to(dev)).cpu()
                thinner_mlp_oups_list.append(thinner_mlp_oups)

            thinner_mlp_oups_list = torch.cat(thinner_mlp_oups_list)
            residual = mlp_oups - thinner_mlp_oups_list
            solution, importance, pred = solve(mlp_inps, residual, args)

            weight = ((pred - pred.mean(0, keepdim=True)) * ( residual - residual.mean(0, keepdim=True))).mean(0) / (pred.std(0) * residual.std(0))
            weight = nn.Sigmoid()((weight - weight.mean()) / 0.05)

            print(layer_index, weight.sort()[0])

            weighted_solution = solution * weight.reshape(1, -1)
            U, S, V = torch.linalg.svd(weighted_solution.to(dev))
            r = int(args.ratio * len(S))
            w1 = U[:, :r] * S[:r].reshape(1, -1)
            w2 = V[:r] / weight.reshape(1, -1).to(dev)

            w1 = w1.to(mlp.gate_proj.weight)
            w2 = w2.to(mlp.gate_proj.weight)

            linear = SVDLinearForWidth(w1=w1, w2=w2)
            branch_mlp = CustomizedMLP(mlp=mlp, linear=linear)
            layer.mlp = branch_mlp
            del U, S, V, w1, w2, linear

        for j in range(len(layer_inputs)):
            with torch.no_grad():
                layer_inputs[j] = \
                    layer(layer_inputs[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),
                          position_ids=position_ids.to(dev))[
                        0].to(layer_inputs).cpu()

        layers[layer_index] = layer.cpu()
        del layer, mlp_inps, mlp_oups
        torch.cuda.empty_cache()

    n_total_params_pruned = sum([p.numel() for p in model.parameters()])
    sparsity = (n_total_params - n_total_params_pruned) / n_total_params
    print('Sparsity: qk: {}, vp: {}, mlp: {}'.format(sparsity_qk, sparsity_vp, sparsity_mlp))
    print('Sparsity: {:.2f}'.format(sparsity))
    torch.cuda.empty_cache()
    return sparsity_qk, sparsity_vp, sparsity_mlp, sparsity


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value