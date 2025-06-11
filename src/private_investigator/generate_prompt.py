
from os import path
from numpy import argsort
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import yaml

from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.models.model_factory import ModelFactory
from .pii_collector import PIICollector


def generate_200(inputs):
    previous_prompt, target, ground_truth, model_args, gpu_id, num_gpus = inputs
    # generate 200 texts using all possible tokens
    # setting
    env_args = EnvArgs(device=f'cuda:{gpu_id}')
    sampling_args = SamplingArgs(N=200, seq_len=256)
    lm_target = ModelFactory.from_model_args(model_args, env_args=env_args).load()
    pii_collector = PIICollector(target)

    # get list of prompts to use for generation
    start = lm_target._tokenizer.vocab_size * gpu_id // num_gpus
    end = lm_target._tokenizer.vocab_size * (gpu_id + 1) // num_gpus
    prompts = [previous_prompt + lm_target._tokenizer.decode(token) for token in range(start, end)]

    # generate texts and count PIIs in the sentences
    counts = []
    for prompt in tqdm(prompts):
        sampling_args.prompt = prompt
        texts = lm_target.generate(sampling_args)
        new_piis = pii_collector.get_pii(texts)
        new_piis = list(set(new_piis))
        piis = []
        for pii in new_piis:
            if pii in ground_truth:
                piis.append(pii)
        counts.append(len(piis))
    return prompts, counts

def generate_2000(inputs):
    top_prompts, target, ground_truth, model_args, gpu_id, num_gpus = inputs
    # generate 2000 texts using top 1% prompts
    # setting
    env_args = EnvArgs(device=f'cuda:{gpu_id}')
    sampling_args = SamplingArgs(N=2000, seq_len=256)
    lm_target = ModelFactory.from_model_args(model_args, env_args=env_args).load()
    pii_collector = PIICollector(target)

    # generate texts and count PIIs in the sentences
    start = len(top_prompts) * gpu_id // num_gpus
    end = len(top_prompts) * (gpu_id + 1) // num_gpus
    counts = []
    for prompt in tqdm(top_prompts[start:end]):
        sampling_args.prompt = prompt
        texts = lm_target.generate(sampling_args)
        new_piis = pii_collector.get_pii(texts)
        new_piis = list(set(new_piis))
        piis = []
        for pii in new_piis:
            if pii in ground_truth:
                piis.append(pii)
        counts.append(len(piis))
    return counts
    

def get_hidden_states(model, prompt):
    with torch.no_grad():
        input_ids = model._tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        hidden_states = model._lm(input_ids[:,-5:], output_hidden_states=True)['hidden_states']
    last_hidden_states = []
    for hidden_state in hidden_states:
        last_hidden_states.append(hidden_state[0,-1,:].to('cpu'))
    return torch.stack(last_hidden_states)


def generate_prompt(previous_prompt, target, ground_truth, model_args, default_path, from_scratch):
    # setting for multi-gpu text generation
    num_gpus = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    path_2000 = default_path + '_2000.yml'
    if path.isfile(path_2000) and not from_scratch:
        with open(path_2000, 'r') as f:
            top_prompts, top_counts = yaml.load(f, Loader=yaml.FullLoader)
    else:
        path_200 = default_path + '_200.yml'
        if path.isfile(path_200) and not from_scratch:
            with open(path_200, 'r') as f:
                prompts, counts = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print('Generate 200 texts using all possible tokens and count piis in the generated texts')
            inputs = [(previous_prompt, target, ground_truth, model_args, i, num_gpus) for i in range(num_gpus)]
            with Pool(processes=num_gpus) as p:
                results = p.map(generate_200, inputs)
            prompts = []
            counts = []
            for p, c in results:
                prompts.extend(p)
                counts.extend(c)
            with open(path_200, 'w') as f:
                yaml.dump([prompts, counts], f)

        # collect top 1% prompts
        top_index = argsort(counts)[::-1][:len(prompts)//100]
        top_prompts = [prompts[i] for i in top_index]

        print('Generate 2000 texts using top 1% prompts and count piis in the generated texts')
        inputs = [(top_prompts, target, ground_truth, model_args, i, num_gpus) for i in range(num_gpus)]
        with Pool(processes=num_gpus) as p:
            results = p.map(generate_2000, inputs)
        top_counts = []
        for c in results:
            top_counts.extend(c)
        assert len(top_prompts) == len(top_counts)
        with open(path_2000, 'w') as f:
            yaml.dump([top_prompts, top_counts], f)


    print('select 20 prompts')
    # select the most promising prompt
    promising_indices = argsort(top_counts)[::-1]
    selected_indices = [promising_indices[0]]

    # get hidden states of prompts
    lm_target = ModelFactory.from_model_args(model_args).load()
    prompt_hiddens = []
    for prompt in tqdm(top_prompts):
        prompt_hiddens.append(get_hidden_states(lm_target, prompt))

    # select 19 prompts sparsely
    for _ in range(19):
        selected_hidden = [prompt_hiddens[i] for i in selected_indices]
        hidden_mean = torch.mean(torch.stack(selected_hidden), 0)
        min_cos_sim = 1
        curr_index = -1
        for i, hidden in enumerate(prompt_hiddens):
            if i in selected_indices:
                continue
            cos_sim = cosine_similarity(hidden_mean, hidden)[-2]
            if min_cos_sim > cos_sim:
                min_cos_sim = cos_sim
                curr_index = i
            elif min_cos_sim + 0.01 >= cos_sim and top_counts[i] > top_counts[curr_index]:
                curr_index = i
        print(min_cos_sim)
        print(top_counts[curr_index])
        selected_indices.append(curr_index)
    generated_prompts = [top_prompts[i] for i in selected_indices]

    return generated_prompts