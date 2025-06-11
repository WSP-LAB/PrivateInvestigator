from os import makedirs, path

from filelock import FileLock
from transformers import HfArgumentParser
import yaml

from carlini.extraction import generation
from carlini.carlini_args import CarliniArgs
from private_investigator.pii_collector import PIICollector


def parse_args():
    parser = HfArgumentParser(CarliniArgs)
    return parser.parse_args_into_dataclasses()

def main(carlini_args):
    # read arguments
    model_name = carlini_args.model_path.split('/')[-1]
    if model_name == '':
        model_name = carlini_args.model_path.split('/')[-2]
    dataset = model_name.split('-')[1]
    if 'dp' in dataset:
        dataset = dataset[:-3]

    # set default path to save results
    makedirs(carlini_args.result_path, exist_ok=True)
    default_path = f'''{model_name}.{carlini_args.method}.temp_{carlini_args.temp}'''
    print(default_path)
    default_path = path.join(carlini_args.result_path, default_path)

    # check whether the previous results exist
    pii_path = f'{default_path}.{carlini_args.target}.yml'
    if not path.isfile(pii_path):

        # check whether the previously generated texts exist
        text_path = f'{default_path}.texts.yml'
        if not path.isfile(text_path):
            # generate texts
            print('[Baseline-Carlini] Generate texts with target model.\n')
            texts = generation(carlini_args, text_path)
            with open(text_path, 'w') as f:
                yaml.dump(texts, f)
        else:
            # load previously generated texts
            with open(text_path, 'r') as f:
                texts = yaml.load(f, Loader=yaml.FullLoader)
        assert(len(texts) == carlini_args.N)

        # collect piis
        print('[Baseline-Carlini] Collect PIIss from generated texts.\n')
        pii_collector = PIICollector(carlini_args.target)
        piis = pii_collector.get_pii(texts)
        if carlini_args.target == 'phone':
            piis = [pii.replace('(','').replace(')','').replace(' ','-').replace('.','-') for pii in piis]
        # store piis
        with open(pii_path, 'w') as f:
            yaml.dump(piis, f)
    else:
        with open(pii_path, 'r') as f:
            piis = yaml.load(f, Loader=yaml.FullLoader)

    piis = list(set(piis))

    # check result
    # load ground truth
    with open(f'piis/{dataset}_{carlini_args.target}_train.yml', 'r') as f:
        ground_truth = yaml.load(f, Loader=yaml.FullLoader)
    # collect successfully extracted PIIs
    succeed = []
    for pii in piis:
        if pii in ground_truth:
            succeed.append(pii)
    print(f'{len(succeed)}')

    lock = FileLock('all_attack_result.txt.lock')
    with lock:
        with open('all_attack_result.txt', 'a') as f:
            f.write(f"Carlini-{carlini_args.method}\t{model_name}\t{carlini_args.target}\t: {len(succeed)}\n")


if __name__ == '__main__':
    main(*parse_args())
