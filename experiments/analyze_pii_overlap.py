from os import path

import yaml

def get_train_piis(piis, ground_truth):
    train_piis = []
    for pii in piis:
        if pii in ground_truth:
            train_piis.append(pii)
    return train_piis

def check_overlap_baseline(model, dataset, target, ground_truth):
    # set paths
    if target == 'name':
        our_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.4.name.yml'
        car_path = f'carlini_result/{model}-{dataset}.topk.temp_1.4.name.yml'
        luk_path = f'lukas_result/{model}-{dataset}.temp_1.4.name.yml'
    else:
        our_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
        car_path = f'carlini_result/{model}-{dataset}.topk.temp_1.{target}.yml'
        luk_path = f'lukas_result/{model}-{dataset}.temp_1.0.{target}.yml'

    # if there are missing results, skip
    if not (path.isfile(our_path) and path.isfile(car_path)
            and path.isfile(luk_path)):
        return ''
    print(f'Checking overlap for {model}-{dataset}-{target}')

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
    car_piis = list(set(car_piis))
    # collect piis that exists in the training data
    car_piis = get_train_piis(car_piis, ground_truth)

    # load lukas result
    with open(luk_path, 'r') as f:
        luk_piis = yaml.load(f, Loader=yaml.FullLoader)
    luk_piis = list(set(luk_piis))
    # collect piis that exists in the training data
    luk_piis = get_train_piis(luk_piis, ground_truth)

    # count piis included in each area in diagram
    ocl = 0
    ol = 0
    co = 0
    o = 0
    for pii in our_piis:
        if pii in luk_piis:
            if pii in car_piis:
                ocl += 1
            else:
                ol +=1
        elif pii in car_piis:
            co += 1
        else:
            o += 1
    lc = -ocl
    for pii in luk_piis:
        if pii in car_piis:
            lc += 1

    # print result
    message = 'ours\tlukas\tcarlini\tcarlini-ours\tours-lukas\tlukas-carlini\tours-carlini-lukas\n'
    message += f'{o}\t{len(luk_piis)-ol-lc-ocl}\t{len(car_piis)-co-lc-ocl}\t{co}\t\t{ol}\t\t{lc}\t\t{ocl}\n'
    print(message)
    return message

def check_overlap_surrogate(model, dataset, target, ground_truth):
    # set paths
    if target == 'name':
        pre_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.4.name.yml'
        fine_path = f'private_investigator_fine/{model}-{dataset}.c_0.25.temp_1.4.name.yml'
    else:
        pre_path = f'private_investigator_pre/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
        fine_path = f'private_investigator_fine/{model}-{dataset}.c_0.25.temp_1.0.{target}.yml'
    # if there are missing results, pass
    if not (path.isfile(pre_path) and path.isfile(fine_path)):
        return ''
    print(f'Checking overlap for {model}-{dataset}-{target}')

    # load private investigator result pre
    with open(pre_path, 'r') as f:
        result = yaml.load(f, Loader=yaml.FullLoader)[0]
    pre_piis = []
    for pii_list in result:
        pre_piis.extend(pii_list)
    pre_piis = list(set(pre_piis))
    # collect piis that exists in the training data
    pre_piis = get_train_piis(pre_piis, ground_truth)

    # load private investigator result fine
    with open(fine_path, 'r') as f:
        result = yaml.load(f, Loader=yaml.FullLoader)[0]
    fine_piis = []
    for pii_list in result:
        fine_piis.extend(pii_list)
    fine_piis = list(set(fine_piis))
    # collect piis that exists in the training data
    fine_piis = get_train_piis(fine_piis, ground_truth)

    # count piis included in each area in diagram
    pf = 0
    for pii in pre_piis:
        if pii in fine_piis:
            pf+=1

    # print result
    message = f'pre\tfine\tpre-fine\n{len(pre_piis)-pf}\t{len(fine_piis)-pf}\t{pf}\n'
    print(message)
    return message

def main():
    all_messages = ''
    # load ground truth
    ground_truth_dict = {}
    for dataset in ['enron', 'trec']:
        for target in ['email', 'phone', 'name']:
            with open(f'piis/{dataset}_{target}_train.yml', 'r') as f:
                ground_truth = yaml.load(f, Loader=yaml.FullLoader)
                ground_truth_dict[dataset + target] = ground_truth

    # count overlaped PIIs between Private Investigator and baselines
    for model in ['gpt2', 'gptneo', 'openelm', 'phi2']:
        for dataset in ['enron', 'trec']:
            for target in ['email', 'phone', 'name']:
                all_messages += check_overlap_baseline(
                    model, dataset, target,
                    ground_truth_dict[dataset + target]
                )

    # count overlaped PIIs between pre and fine surrogate models
    for target in ['email', 'phone', 'name']:
        all_messages += check_overlap_surrogate(
            'gptneo', 'enron', target, ground_truth_dict['enron' + target]
        )


    with open('pii_overlap.txt', 'w') as f:
        f.write(all_messages)


if __name__ == '__main__':
    main()
