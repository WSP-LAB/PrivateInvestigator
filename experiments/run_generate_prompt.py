from os import makedirs

import transformers
import yaml
from numpy import argsort

from private_investigator.arguments.prompt_args import PromptArgs
from private_investigator.generate_prompt import generate_prompt

from pii_leakage.arguments.model_args import ModelArgs


def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArgs, PromptArgs)
    )
    return parser.parse_args_into_dataclasses()


def main(model_args, prompt_args):
    makedirs('text_generation', exist_ok=True)
    # read arguments
    if model_args.model_ckpt is not None:
        model_name = model_args.model_ckpt.split('/')[-1]
        if model_name == '':
            model_name = model_args.model_ckpt.split('/')[-2]
        dataset = model_name.split('-')[1]
    else:
        model_name = 'gptneo'
        dataset = 'pile'

    # load previous prompt if the prompt length is longer than 1
    if prompt_args.prompt_len > 1:
        if prompt_args.surrogate == 'pre':
            with open(f'prompts/pre_{prompt_args.target}_{prompt_args.prompt_len}.yml', 'r') as f:
                previous_prompt = yaml.load(f, Loader=yaml.FullLoader)[0]
        else:
            with open(f'prompts/fine_{dataset}_{prompt_args.target}_{prompt_args.prompt_len}.yml', 'r') as f:
                previous_prompt = yaml.load(f, Loader=yaml.FullLoader)[0]
    else:
        previous_prompt = ''

    # load ground truth
    with open(f'piis/{dataset}_{prompt_args.target}_train.yml', 'r') as f:
        ground_truth = yaml.load(f, Loader=yaml.FullLoader)

    # generate prompts
    default_path = f'text_generation/{model_name}_{prompt_args.target}_{prompt_args.prompt_len}'
    prompts = generate_prompt(
        previous_prompt, prompt_args.target, ground_truth, model_args,
        default_path, prompt_args.from_scratch
    )
    print(f'generated prompts {prompts}')

    # store prompts
    if prompt_args.surrogate == 'pre':
        with open(f'prompts/pre_{prompt_args.target}_{prompt_args.prompt_len}.yml', 'w') as f:
            yaml.dump(prompts, f)
    else:
        with open(f'prompts/fine_{dataset}_{prompt_args.target}_{prompt_args.prompt_len}.yml', 'w') as f:
            yaml.dump(prompts, f)

if __name__ == '__main__':
    main(*parse_args())
