from os import path

from tqdm import tqdm
import numpy as np
import transformers
import yaml

from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.models.model_factory import ModelFactory

def get_train_piis(piis, ground_truth):
    train_piis = []
    for pii in piis:
        if pii in ground_truth:
            train_piis.append(pii)
    return train_piis

def compare_pii_perplexity(model, dataset, target):
    # set paths
    our_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
    car_path = f'carlini_result/{model}-{dataset}.topk.temp_1.{target}.yml'
    luk_path = f'lukas_result/{model}-{dataset}.temp_1.0.{target}.yml'

    # if there are missing results, skip
    if not (path.isfile(our_path) and path.isfile(car_path)
            and path.isfile(luk_path)):
        return ''
    print(f'Computing PII perplexity for {model}-{dataset}-{target}')

    # load ground truth
    with open(f'piis/{dataset}_{target}_train.yml', 'r') as f:
        ground_truth = yaml.load(f, Loader=yaml.FullLoader)

    # load private investigator result
    with open(our_path, 'r') as f:
        result = yaml.load(f, Loader=yaml.FullLoader)[0]
    our_piis = []
    for pii_list in result:
        our_piis.extend(pii_list)
    our_piis = list(set(our_piis))
    # collect piis that exists in the training data
    our_piis = get_train_piis(our_piis, ground_truth)

    # load carlini result
    with open(car_path, 'r') as f:
        car_piis = yaml.load(f, Loader=yaml.FullLoader)
    # load lukas result
    with open(luk_path, 'r') as f:
        luk_piis = yaml.load(f, Loader=yaml.FullLoader)
    # collect baseline piis that exists in the training data
    base_piis = car_piis + luk_piis
    base_piis = list(set(base_piis))
    base_piis = get_train_piis(base_piis, ground_truth)

    # load model
    model_args = ModelArgs(architecture=model,
                           model_ckpt=f'weights/{model}-{dataset}')
    lm = ModelFactory.from_model_args(model_args).load()

    # get our exclusize piis
    our_piis_exclusive = []
    for pii in our_piis:
        if pii not in base_piis:
            our_piis_exclusive.append(pii)
    # compute perplexity
    our_ppls = []
    for pii in tqdm(our_piis_exclusive):
        # pii = lm._tokenizer.bos_token + pii
        our_ppls.append(lm.perplexity(pii, verbose=False))
    print('ours', np.mean(our_ppls))

    # get baseline exclusive piis
    base_piis_exclusive = []
    for pii in base_piis:
        if pii not in our_piis:
            base_piis_exclusive.append(pii)
    # compute perplexity
    base_ppls = []
    for pii in tqdm(base_piis_exclusive):
        # pii = lm._tokenizer.bos_token + pii
        base_ppls.append(lm.perplexity(pii, verbose=False))
    print('baselines', np.mean(base_ppls))

    message = f'{model}\t{dataset}\t{target}\n'
    message += f'ours: {np.mean(our_ppls)}, baselines: {np.mean(base_ppls)}\n'
    return message

def main():
    all_messages = ''
    for target in ['email', 'phone']:
        for dataset in ['enron', 'trec']:
            all_messages += compare_pii_perplexity('gptneo', dataset, target)
    with open('perplexity.txt', 'w') as f:
        f.write(all_messages)

if __name__ == '__main__':
    main()
