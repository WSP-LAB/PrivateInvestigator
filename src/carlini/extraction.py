import logging
logging.basicConfig(level='ERROR')

import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import yaml

def load_gpt2(model_path):
    print("Loading GPT2...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left" 
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True).to('cuda')
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    model.resize_token_embeddings(len(tokenizer))
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
    return tokenizer, model

def load_gptneo(model_path):
    print("Loading GPT-NEO...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
    tokenizer.padding_side = "left" 
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True).to('cuda')
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    model.resize_token_embeddings(len(tokenizer))
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    return tokenizer, model

def load_openelm(model_path):
    print("Loading OpenELM...")
    from huggingface_hub import login
    login(token='Model_Access_Token')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True, model_max_length=4096)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left" 
    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, trust_remote_code=True).eval().to('cuda')
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def load_phi2(model_path):
    print("Loading PHI2...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True, model_max_length=4096)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left" 
    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, trust_remote_code=True, torch_dtype=torch.float16).eval().to('cuda')
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng



def generation(args, result_path):
    print(f"using device: cuda")

    if args.method == 'internet':
        print("Loading common crawl...")
        cc = parse_commoncrawl(args.wet_file)

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    # load tokenizer and model
    if 'gpt2' in args.model_path:
        tokenizer, model = load_gpt2(args.model_path)
    elif 'gptneo' in args.model_path:
        tokenizer, model = load_gptneo(args.model_path)
    elif 'openelm' in args.model_path:
        tokenizer, model = load_openelm(args.model_path)
    elif 'phi2' in args.model_path:
        tokenizer, model = load_phi2(args.model_path)
    else:
        raise ValueError(args.model_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    all_texts = []
    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N, desc="Generating texts") as pbar:
        for i in range(num_batches):
            # encode the prompts
            if args.method == 'internet':
                # pick a random 10-token prompt in common crawl 
                input_len = random.randint(5,10)
                input_ids = []
                attention_mask = []
                prompt_texts = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    index = random.randint(0, len(cc)-100)
                    prompt = " ".join(cc[index:index+100].split(" ")[1:])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                    prompt = inputs['input_ids'][0]

                    if len(prompt) > input_len:
                        input_ids.append(prompt[:input_len])
                        attention_mask.append(inputs['attention_mask'][0][:input_len])
                        prompt_texts.append(tokenizer.decode(prompt[:input_len]))

                inputs = {'input_ids': torch.stack(input_ids), 
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                prompts = [tokenizer.bos_token] * args.batch_size
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                input_len = len(inputs[0])

            # batch generation
            if args.method == 'temperature':
                sequences = inputs['input_ids'].to('cuda')
                for i in range(20):
                    attention_mask = torch.ones_like(sequences, dtype=torch.int, device=torch.device('cuda'))
                    temperature = 10 - (10 - args.temp) * i / 19 
                    sequences = model.generate(
                        input_ids=sequences,
                        attention_mask=attention_mask,
                        max_length=input_len + i + 1,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k, 
                        top_p=1.0
                    )

                attention_mask = torch.ones_like(sequences, dtype=torch.int, device=torch.device('cuda'))

                output_sequences = model.generate(
                    input_ids=sequences,
                    attention_mask=attention_mask,
                    max_length=input_len + seq_len,
                    do_sample=True,
                    temperature=args.temp,
                    top_k=top_k, 
                    top_p=1.0
                )
            else:
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'].to('cuda'),
                    attention_mask=inputs['attention_mask'].to('cuda'),
                    max_length=input_len + seq_len,
                    do_sample=True, 
                    temperature=args.temp,
                    top_k=top_k, 
                    top_p=1.0
                )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            all_texts.extend(texts)

            pbar.update(args.batch_size)

    with open(result_path, 'w') as f:
        yaml.dump(all_texts, f)

    return all_texts
