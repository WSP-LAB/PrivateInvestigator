import csv
from os import makedirs, path

import yaml
import matplotlib.pyplot as plt


def get_train_piis(piis, ground_truth):
    train_piis = []
    for pii in piis:
        if pii in ground_truth:
            train_piis.append(pii)
    return train_piis


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars

    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
        bars.append(bar[0])

    if legend:
        ax.legend(bars, data.keys())


def get_bars(counts, bins):
    result = [0] * len(bins)
    for count in counts:
        for i, bin_border in enumerate(bins):
            if count <= bin_border:
                result[i] += 1
                break
    return result


def check_num_duplicate(model, dataset):
    carlini_pii_dup_count = []
    lukas_pii_dup_count = []
    ours_pii_dup_count = []

    for target in ['email', 'phone']:
        # set paths
        our_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
        car_path = f'carlini_result/{model}-{dataset}.topk.temp_1.{target}.yml'
        luk_path = f'lukas_result/{model}-{dataset}.temp_1.0.{target}.yml'

        # if there are missing results, skip
        if not (path.isfile(our_path) and path.isfile(car_path)
                and path.isfile(luk_path)):
            return

        # load ground truth
        with open(f'piis/{dataset}_{target}_train_dict.yml', 'r') as f:
            counts = yaml.load(f, Loader=yaml.FullLoader)

        # load carlini result
        with open(car_path, 'r') as f:
            car_piis = yaml.load(f, Loader=yaml.FullLoader)
        car_piis = list(set(car_piis))
        car_piis = get_train_piis(car_piis, counts)

        # load lukas result
        with open(luk_path, 'r') as f:
            luk_piis = yaml.load(f, Loader=yaml.FullLoader)
        luk_piis = list(set(luk_piis))
        luk_piis = get_train_piis(luk_piis, counts)

        # load private investigator result
        with open(our_path, 'r') as f:
            result = yaml.load(f, Loader=yaml.FullLoader)[0]
        our_piis = []
        for pii_list in result:
            our_piis.extend(pii_list)
        our_piis = list(set(our_piis))
        our_piis = get_train_piis(our_piis, counts)

        # get exclusive piis
        car_piis_exclusive = []
        for pii in car_piis:
            if pii not in luk_piis and pii not in our_piis:
                car_piis_exclusive.append(pii)
        luk_piis_exclusive = []
        for pii in luk_piis:
            if pii not in car_piis and pii not in our_piis:
                luk_piis_exclusive.append(pii)
        our_piis_exclusive = []
        for pii in our_piis:
            if pii not in car_piis and pii not in luk_piis:
                our_piis_exclusive.append(pii)

        # count number of duplication
        for pii in car_piis_exclusive:
            carlini_pii_dup_count.append(counts[pii])
        for pii in luk_piis_exclusive:
            lukas_pii_dup_count.append(counts[pii])
        for pii in our_piis_exclusive:
            ours_pii_dup_count.append(counts[pii])

    # count piis included in each bin
    bins = [1, 5, 10, 50, 100, 5000]
    carlini_bar = get_bars(carlini_pii_dup_count, bins)
    lukas_bar = get_bars(lukas_pii_dup_count, bins)
    ours_bar = get_bars(ours_pii_dup_count, bins)

    # draw bar graph
    data = {
        "carlini": carlini_bar,
        "lukas": lukas_bar,
        "ours": ours_bar,
    }
    plt.clf()
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.savefig(f'figure/Duplication_{model}_{dataset}_{target}.png')

    # save result in csv file
    with open(f'csv/duplicate_{model}_{dataset}.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(carlini_bar)):
            writer.writerow([ours_bar[i], carlini_bar[i], lukas_bar[i]])


def main():
    makedirs('figure', exist_ok=True)
    makedirs('csv', exist_ok=True)
    check_num_duplicate('gptneo', 'enron')
    check_num_duplicate('gptneo', 'trec')


if __name__ == '__main__':
    main()
