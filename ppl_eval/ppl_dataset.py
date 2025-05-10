'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext2():
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_dataset("arrow", data_files={"test": "/userhome/home/hejiujun/project/LLM/LoRAPrune/data/wikitext/wikitext-train.arrow"}, split="train")
    # testdata = load_dataset("arrow", data_files={"test": "/userhome/home/hejiujun/project/LLM/LoRAPrune/data/wikitext/wikitext-test.arrow"}, split="test")
    return traindata, testdata

def get_ptb():
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       
def get_loaders(name, tokenizer, seq_len=128, batch_size = 8):
    if 'wikitext2' in name:
        from datasets import load_dataset, load_from_disk
        all_data = load_dataset(
            # '/userhome/home/hejiujun/project/LLM/LoRAPrune/data/wikitext/'
            '/userhome/home/hejiujun/project/LLMs/LoRAP/data/wikitext/'
        )
        train_data, test_data = all_data['train'], all_data['test']
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        train_data, test_data = get_ptb()
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader