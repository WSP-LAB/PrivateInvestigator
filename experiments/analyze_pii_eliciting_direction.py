import csv
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import transformers
import yaml

from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.models.model_factory import ModelFactory

def parse_args():
    parser = transformers.HfArgumentParser(
        (EnvArgs, ModelArgs)
    )
    return parser.parse_args_into_dataclasses()


def get_hidden_states(model, prompt):
    with torch.no_grad():
        input_ids = model._tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        hidden_states = model._lm(input_ids[:,-1024:], output_hidden_states=True)['hidden_states']
    last_hidden_states = []
    for hidden_state in hidden_states:
        last_hidden_states.append(hidden_state[0,-1,:].to('cpu'))
    return torch.stack(last_hidden_states)


def get_all_hidden(lm_target):
    print('get hidden states of all tokens')
    vocab_size = lm_target._tokenizer.vocab_size
    hidden = []
    for i in tqdm(range(vocab_size)):
        prompt = lm_target._tokenizer.decode(i)
        hidden.append(get_hidden_states(lm_target, prompt))
    hidden = torch.stack(hidden)
    return hidden


def get_our_hidden(lm_target, model_name, target):
    print('get hidden states of our prompts')
    # load prompts
    if 'enron' in model_name:
        prompt_path = f'prompts/fine_trec_{target}_1.yml'
    else:
        prompt_path = f'prompts/fine_enron_{target}_1.yml'
    with open(prompt_path, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)

    # get hidden vectors
    our_hidden = []
    for prompt in prompts:
        our_hidden.append(get_hidden_states(lm_target, prompt))
    our_hidden = torch.stack(our_hidden)

    return our_hidden


def get_goodnbad_hidden(lm_target, model_name, target):
    print('get bad hidden')
    # load prompts and corresponding pii counts
    with open(f'text_generation/{model_name}_{target}_1_200.yml', 'r') as f:
        prompts, counts = yaml.load(f, Loader=yaml.FullLoader)
    assert(len(prompts) == len(counts))

    # get least performing 100 prompts
    sorted_indice = np.argsort(counts)
    bad_prompts = []
    for index in sorted_indice[:100]:
        bad_prompts.append(prompts[index])
    # get mean hidden vector
    bad_hidden = []
    for prompt in bad_prompts:
        bad_hidden.append(get_hidden_states(lm_target, prompt))
    bad_hidden = torch.mean(torch.stack(bad_hidden), 0)

    print('get good hidden')
    # load prompts and corresponding pii counts
    with open(f'text_generation/{model_name}_{target}_1_2000.yml', 'r') as f:
        top_prompts, top_counts = yaml.load(f, Loader=yaml.FullLoader)
    assert(len(top_prompts) == len(top_counts))

    # get best performing 100 prompts
    sorted_indice = np.argsort(top_counts)[::-1]
    good_prompts = []
    for index in sorted_indice[:100]:
        good_prompts.append(top_prompts[index])
    # get mean hidden vector
    good_hidden = []
    for prompt in good_prompts:
        good_hidden.append(get_hidden_states(lm_target, prompt))
    good_hidden = torch.mean(torch.stack(good_hidden), 0)

    return good_hidden, bad_hidden


def main(model_args, prompt_args):
    # set directories
    makedirs('figure', exist_ok=True)
    makedirs('csv', exist_ok=True)

    # load model
    model_name = model_args.model_ckpt.split('/')[-1]
    if model_name == '':
        model_name = model_args.model_ckpt.split('/')[-2]
    lm_target = ModelFactory.from_model_args(model_args).load(verbose=True)

    # get context vector
    good_hidden_true, bad_hidden_true = get_goodnbad_hidden(lm_target, model_name, prompt_args.target)
    context = good_hidden_true-bad_hidden_true
    max_sim = cosine_similarity(good_hidden_true, context)
    min_sim = cosine_similarity(bad_hidden_true, context)

    # get hidden states of our prompts
    our_hiddens = get_our_hidden(lm_target, model_name, prompt_args.target)
    # calculate cosine similarity
    our_sims = []
    for hidden in our_hiddens:
        our_sims.append(cosine_similarity(hidden, context))
    our_sims = torch.stack(our_sims)
    our_sims = torch.quantile(our_sims, torch.tensor([0.25, 0.5, 0.75]), dim=0)

    # get hidden states of all prompts
    all_hiddens = get_all_hidden(lm_target)
    # calculate cosine similarity
    all_sims = []
    for hidden in all_hiddens:
        all_sims.append(cosine_similarity(hidden, context))
    all_sims = torch.stack(all_sims)
    all_sims = torch.quantile(all_sims, torch.tensor([0.25, 0.5, 0.75]), dim=0)

    # normalize
    for i in range(len(context)):
        our_sims[:,i] = (our_sims[:,i] - min_sim[i]) / (max_sim[i] - min_sim[i])
        all_sims[:,i] = (all_sims[:,i] - min_sim[i]) / (max_sim[i] - min_sim[i])

    # draw figure
    x = list(range(our_sims.shape[1]))
    plt.plot(x, our_sims[1], label='our')
    plt.plot(x, all_sims[1], label='all')
    plt.legend()
    plt.savefig(f'figure/Pii-eliciting-derections_{model_name}_{prompt_args.target}.png')

    with open(f'csv/Pii-eliciting-derections_{model_name}_{prompt_args.target}.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(our_sims.shape[1]):
            writer.writerow([
                our_sims[0,i].item(), our_sims[1,i].item(), our_sims[2,i].item(),
                all_sims[0,i].item(), all_sims[1,i].item(), all_sims[2,i].item()
            ])


if __name__ == '__main__':
    main(*parse_args())
