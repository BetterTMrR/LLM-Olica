## Olica: Efficient Structured Pruning of Large Language Models without Retraining (ICML 2025)

Olica requires only hundreds of samples, several minutes of runtime, and a single 16GB GPU to prune a LLaMA-7B model. It can even prune a LLaMA-30B model on the same 16GB GPU with less than an hour of runtime.

## Getting Started

**Environment setup**

```sh
pip install -r requirements.txt
```

Warning: ``transformers==4.31.0`` must be forcibly required.

### Pruning with Olica

```sh
CUDA_VISIBLE_DEVICES=0 python model_pruning.py \
--base_model 7b \
--datasets bookcorpus+alpaca \
--num_samples 128 \
--seqlen 128 \
--percdamp 0.5 \
--mlp_num 6 \
--sparsity 0.2 \
--ratio 0.03 \
--k 3 \
--save_dir ./pruned_models
```

which takes approximately 7 minutes on a single NVIDIA GeForce RTX 4090 GPU.

Arguments:

- ``--base_model``: The size of LLaMA model.
- ``--datasets``: Calibration datasets for sampling.
- ``--num_samples``: Specify the number of calibration samples drawn from each dataset.
- ``--seqlen``: Sequence length of each calibration sample.
- ``--percdamp``: Can be seen as the penalty strength $\lambda$ in the regression loss, i.e., Eq.(7).
- ``--mlp_num``: Number of calibrated MLP layers.
- ``--sparsity``: Sparsity ratio of the pruned model.
- ``--ratio``: Low-rank ratio of SVD in the linear calibration.
- ``--k``: Ratio of QK and VO parameters.
- ``--save_dir``: The director of saving the pruned model.

### Evaluate the Pruned Model

```sh
CUDA_VISIBLE_DEVICES=0 python model_evaluate.py \
--base_mode 7b \
--sparsity 0.2 \
--tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
--save_dir ./pruned_models
```

Optionally, you can run `bash scripts/pruning_and_eval_7b.sh` and `bash scripts/pruning_and_eval_13b.sh`.

## Acknowledgments

Our code is partially based on the following projects. We thank their awesome works.

- [LLM-Pruner](https://github.com/horseee/LLM-Pruner)
- [gptq](https://github.com/ist-daslab/gptq)
- [sparsegpt](https://github.com/ist-daslab/sparsegpt)
- [SlimGPT](https://openreview.net/forum?id=MxF0IKJtKW)

## Citation

If you find our work helpful, please consider citing the following paper.

```
@inproceedings{he2025olica,
  title={Olica: Efficient Structured Pruning of Large Language Models without Retraining},
  author={Jiujun He and Huazhen Lin},
  booktitle={ICML},
  year={2025}
}
```
