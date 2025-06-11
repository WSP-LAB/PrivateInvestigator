from os import makedirs, path

from filelock import FileLock
from transformers import HfArgumentParser
import yaml

from private_investigator.pii_collector import PIICollector
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.lukas_args import LukasArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.models.model_factory import ModelFactory

def parse_args():
    parser = HfArgumentParser(
        (EnvArgs, LukasArgs, ModelArgs, SamplingArgs)
    )
    return parser.parse_args_into_dataclasses()


def main(env_args, lukas_args, model_args, sampling_args):
    # read arguments
    model_name = model_args.model_ckpt.split('/')[-1]
    if model_name == '':
        model_name = model_args.model_ckpt.split('/')[-2]
    dataset = model_name.split('-')[1]
    if 'dp' in dataset:
        dataset = dataset[:-3]

    # set default path to save results
    makedirs(lukas_args.result_path, exist_ok=True)
    default_path = path.join(lukas_args.result_path,
                                f'{model_name}.temp_{sampling_args.temp}')

    # check whether the previous results exist
    pii_path = default_path + f'.{lukas_args.target}.yml'
    if not path.isfile(pii_path):

        # check whether the previously generated texts exist
        text_path = default_path + '.texts.yml'
        if not path.isfile(text_path):
            # generate texts
            print('[Baseline-Lukas] Generate texts with target model.\n')
            lm_target = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
            sampling_args.seq_len = 256
            sampling_args.generate_verbose = True
            generated_text = lm_target.generate(sampling_args)
            del(lm_target)
            texts = [str(x) for x in generated_text]
            with open(text_path, 'w') as f:
                yaml.dump(texts, f)
        else:
            # load previously generated texts
            with open(text_path, 'r') as f:
                texts = yaml.load(f, Loader=yaml.FullLoader)
        assert(len(texts) == sampling_args.N)

        # collect piis       
        print('[Baseline-Lukas] Collect PIIss from generated texts.\n')
        pii_collector = PIICollector(lukas_args.target)
        piis = pii_collector.get_pii(texts)
        if lukas_args.target == 'phone':
            piis = [pii.replace('(','').replace(')','').replace(' ','-').replace('.','-') for pii in piis]
        with open(pii_path, 'w') as f:
                yaml.dump(piis, f)
    else:
        with open(pii_path, 'r') as f:
            piis = yaml.load(f, Loader=yaml.FullLoader)

    piis = list(set(piis))

    # check result
    # load ground truth
    with open(f'piis/{dataset}_{lukas_args.target}_train.yml', 'r') as f:
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
            f.write(f"Lukas\t\t\t{model_name}\t{lukas_args.target}\t: {len(succeed)}\n")


if __name__ == '__main__':
    main(*parse_args())
