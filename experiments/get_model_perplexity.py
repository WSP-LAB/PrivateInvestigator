import argparse

import numpy as np
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, deepspeed

# import deepspeed

from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.model_factory import ModelFactory

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs, EnvArgs, TrainingArguments))


    return parser.parse_args_into_dataclasses()

def main():
    parser = transformers.HfArgumentParser((ModelArgs, EnvArgs))
    model_args, env_args = parser.parse_args_into_dataclasses()

    ## Set arguments
    model_name = model_args.model_ckpt.split('/')[-1]
    if model_name == '':
        model_name = model_args.model_ckpt.split('/')[-2]
    dataset = model_name.split('-')[1]
    if 'dp' in dataset:
        dataset = dataset[:-3]
    # env_args.device = 'cpu'
    dataset_args = DatasetArgs()
    dataset_args.dataset_path = f'../src/pii_leakage/extern/{dataset}'

    eval_dataset = DatasetFactory.from_dataset_args(
        dataset_args.set_split("train"),
        env_args=env_args
    )
    # with deepspeed.zero.Init():
    # model_args.model_ckpt = args.model_path
    print(model_args.architecture, dataset_args.dataset_path, model_args.model_ckpt)

    target_lm = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
    ppl = target_lm.perplexity(eval_dataset['text'])
    print(ppl)

    model_args.model_ckpt = None
    target_lm = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
    ppl = target_lm.perplexity(eval_dataset['text'])
    print(ppl)


if __name__ == '__main__':
    main()
