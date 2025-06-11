import csv
from os import makedirs, path

import matplotlib.pyplot as plt
import yaml

from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.models.model_factory import ModelFactory


def get_token_counts(piis, token_counts_dict):
    counts = []
    for pii in piis:
        c = token_counts_dict.get(pii, None)
        if c:
            counts.append(c)
    return counts


def get_train_piis(piis, ground_truth):
    train_piis = []
    for pii in piis:
        if pii in ground_truth:
            train_piis.append(pii)
    return train_piis


def compare_pii_length(model, dataset, target='email'):
    # set paths
    pre_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
    dp_path = f'private_investigator_pre/{model}-{dataset}_dp.temp_1.0.{target}.yml'

    # if there are missing results, skip
    if not (path.isfile(pre_path) and path.isfile(dp_path)):
        return

    # load model
    model_args = ModelArgs(architecture=model)
    lm = ModelFactory.from_model_args(model_args).load()

    # load train piis
    with open(f'piis/{dataset}_{target}_train.yml', 'r') as f:
        train_piis = yaml.load(f, Loader=yaml.FullLoader)
    # count tokens
    token_counts = {}
    for pii in train_piis:
        token_counts[pii] = len(lm._tokenizer.encode(pii))
    train_counts = list(token_counts.values())

    # load private investigator result
    with open(pre_path, 'r') as f:
        result = yaml.load(f, Loader=yaml.FullLoader)[0]
    pre_piis = []
    for pii_list in result:
        pre_piis.extend(pii_list)
    pre_piis = list(set(pre_piis))
    # get token counts
    pre_counts = get_token_counts(pre_piis, token_counts)
    
    # load private investigator result dp
    with open(dp_path, 'r') as f:
        result = yaml.load(f, Loader=yaml.FullLoader)[0]
    dp_piis = []
    for pii_list in result:
        dp_piis.extend(pii_list)
    dp_piis = list(set(dp_piis))
    # get token counts
    dp_counts = get_token_counts(dp_piis, token_counts)

    # draw graph
    xs = list(range(min(train_counts), max(train_counts)+1))
    y_train = []
    y_pre = []
    y_dp = []
    for x in xs:
        y_train.append(train_counts.count(x))
        y_pre.append(pre_counts.count(x))
        y_dp.append(dp_counts.count(x))

    max_index = 43

    print(xs[:max_index])
    print(y_train[:max_index])
    print(y_pre[:max_index])
    print(y_dp[:max_index])


    plt.figure(0)
    plt.plot(xs[:max_index], y_train[:max_index], label='train')
    plt.plot(xs[:max_index], y_pre[:max_index], label='ours')
    plt.plot(xs[:max_index], y_dp[:max_index], label='ours(dp)')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'figure/PII_length_{model}_{dataset}_{target}.png')

    with open(f'csv/PII_length_{model}_{dataset}_{target}.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(max_index):
            writer.writerow([xs[i], y_train[i], y_pre[i], y_dp[i]])


def main():
    makedirs('figure', exist_ok=True)
    makedirs('csv', exist_ok=True)
    compare_pii_length('gptneo', 'enron')
    compare_pii_length('gptneo', 'trec')


if __name__ == '__main__':
    main()
