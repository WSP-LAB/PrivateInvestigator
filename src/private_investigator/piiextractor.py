from tqdm import tqdm
import numpy as np
import torch

from .pii_collector import PIICollector

class PIIExtractor():
    def __init__(self, prompts, lm_target, extract_args, prompt_args, sampling_args):
        self.prompts = prompts
        self.lm_target = lm_target
        self.sampling_args = sampling_args
        # initialize
        self.iter = 0
        self.num_selected = np.zeros(len(self.prompts), dtype=np.short)
        self.c = extract_args.c
        self.exploit_score = np.zeros(len(self.prompts))
        self.pii_collector = PIICollector(prompt_args.target)
        self.prompt_texts = [[] for _ in range(len(self.prompts))]
        self.prompt_ppl = [[] for _ in range(len(self.prompts))]
        self.all_piis = []

    def update_score(self, prompt_index, texts):
        # count piis in texts
        piis = self.pii_collector.get_pii(texts)

        self.all_piis.append(piis)

        # get perplexity ratio of texts
        ppl = self.lm_target.perplexity(piis, return_as_list=True, apply_exp=False, verbose=False)
        ppl = - torch.mean(ppl).item()
        self.prompt_ppl[prompt_index].append(ppl)

        if self.iter >= len(self.prompts):
            ppl_list = np.empty(len(self.exploit_score))
            for i in range(len(self.exploit_score)):
                mean_ppl = np.mean(self.prompt_ppl[i])
                ppl_list[i] = mean_ppl
            min_ppl = min(ppl_list)
            max_ppl = max(ppl_list)
            self.exploit_score = (ppl_list-min_ppl) / (max_ppl-min_ppl)
    def select_prompt(self):
        never_chosen = np.where(self.num_selected == 0)[0]
        # select prompts which are never selected
        if len(never_chosen) > 0:
            index = never_chosen[0]
        else:
            # give higher score to pompts which are less selected
            score_explore = (np.log(self.iter) / self.num_selected) ** 0.5
            # Add two scores with balance factor c
            score = self.exploit_score + self.c * score_explore
            # select prompt wiht the maximum score
            index = np.argmax(score)
        return index

    def run(self, n):
        for _ in tqdm(range(n), desc='Private Investigator attack campaign'):
            # select prompt
            prompt_index = self.select_prompt()
            prompt = self.prompts[prompt_index]
            self.iter += 1
            self.num_selected[prompt_index] += 1
            # generate text
            self.sampling_args.prompt = prompt
            texts = self.lm_target.generate(self.sampling_args)
            texts = [text[len(prompt):] for text in texts]
            self.prompt_texts[prompt_index].extend(texts)
            # update exploit score
            self.update_score(prompt_index, texts)

        print("Exploit scores")
        print(self.exploit_score)

        num_all_texts = sum([len(texts)for texts in self.prompt_texts])
        print(f'Total # of texts: {num_all_texts}')

        return self.all_piis, self.prompt_texts, self.num_selected
