from os import makedirs, path

from filelock import FileLock
from transformers import HfArgumentParser
import yaml  

from private_investigator.arguments.extract_args import ExtractArgs
from private_investigator.arguments.prompt_args import PromptArgs
from private_investigator.piiextractor import PIIExtractor

from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.models.model_factory import ModelFactory


def parse_args():
    parser = HfArgumentParser(
        (EnvArgs, ModelArgs, ExtractArgs, PromptArgs, SamplingArgs)
    )
    return parser.parse_args_into_dataclasses()


def main(env_args, model_args, extract_args, prompt_args, sampling_args):
    # read arguments
    model_name = model_args.model_ckpt.split('/')[-1]
    if model_name == '':
        model_name = model_args.model_ckpt.split('/')[-2]
    dataset = model_name.split('-')[1]
    if 'dp' in dataset:
        dataset = dataset[:-3]

    # set default path to save results
    if extract_args.result_path is None:
        extract_args.result_path = f'private_investigator_{prompt_args.surrogate}'
    makedirs(extract_args.result_path, exist_ok=True)
    result_path = path.join(extract_args.result_path,
        f'{model_name}.c_{extract_args.c}.temp_{sampling_args.temp}.{prompt_args.target}.yml')

    # check whether the previous results exist
    if path.isfile(result_path):
        with open(result_path, 'r') as f:
            all_piis = yaml.load(f, Loader=yaml.FullLoader)[0]
        piis = []
        for pii_list in all_piis:
            piis.extend(pii_list)
    else:
        # set arguments
        sampling_args.N = 2000

        # load prompts
        if prompt_args.surrogate == 'pre':
            prompt_path = f'prompts/pre_{prompt_args.target}_{prompt_args.prompt_len}.yml'
        elif prompt_args.surrogate == 'fine':
            if 'enron' in dataset:
                prompt_path = f'prompts/fine_trec_{prompt_args.target}_{prompt_args.prompt_len}.yml'
            else:
                prompt_path = f'prompts/fine_enron_{prompt_args.target}_{prompt_args.prompt_len}.yml'
        else:
            raise ValueError(prompt_args.surrogate)
        with open(prompt_path, 'r') as f:
            prompts = yaml.load(f, Loader=yaml.FullLoader)

        # load model
        lm_target = ModelFactory.from_model_args(model_args, env_args=env_args).load()
        extractor = PIIExtractor(prompts, lm_target, extract_args, prompt_args, sampling_args)

        # run pii extractor
        all_piis, prompt_texts, n_p = extractor.run(extract_args.n)
        if prompt_args.target == 'phone':
            all_piis = [[pii.replace('(','').replace(')','').replace(' ','-').replace('.','-') for pii in pii_list] for pii_list in all_piis]

        # store found piis
        with open(result_path, 'w') as f:
            yaml.dump([all_piis, prompts, n_p.tolist(), prompt_texts], f)
        piis = []
        for pii_list in all_piis:
            piis.extend(pii_list)

    # check result
    # load ground truth
    with open(f'piis/{dataset}_{prompt_args.target}_train.yml', 'r') as f:
        ground_truth = yaml.load(f, Loader=yaml.FullLoader)
    # collect successfully extracted PIIs
    piis = list(set(piis))
    succeed = []
    for pii in piis:
        if pii in ground_truth:
            succeed.append(pii)
    print(f'{len(succeed)}')

    lock = FileLock('all_attack_result.txt.lock')
    with lock:
        with open('all_attack_result.txt', 'a') as f:
            f.write(f'Private Investigator-{prompt_args.surrogate}\t'
                    f'{model_name}\t{prompt_args.target}\t: {len(succeed)}\n')



if __name__ == '__main__':
    main(*parse_args())
