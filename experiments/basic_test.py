import torch
import transformers
from tqdm import tqdm

from carlini.extraction import generation

from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.model_factory import ModelFactory

from private_investigator.generate_prompt import generate_prompt
from private_investigator.piiextractor import PIIExtractor



def main():
    model_args = ModelArgs(
        architecture='openelm', model_ckpt='weights/openelm-trec'
    )


    dataset_args = DatasetArgs()

    dataset_args.dataset_path = f'../src/pii_leakage/extern/trec'
    trec_dataset = DatasetFactory.from_dataset_args(
        dataset_args.set_split("train")
    )


    target_lm = ModelFactory.from_model_args(model_args).load(verbose=True)
    ppl = target_lm.perplexity(trec_dataset['text'][:1000])
    print(ppl)


if __name__ == '__main__':
    main()
