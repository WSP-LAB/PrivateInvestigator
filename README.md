Private Investigator
========

Private Investigator is an attack framework designed to optimize
prompts for querying a target language model to extract PII
used for its fine-tuning process.
Private Investigator outperforms existing attacks in extracting PII items.
For more details, please refer to our paper,
"Private Investigator: Extracting Personally Identifiable Information
from Large Language Models Using Optimized Prompts", which is accepted in USENIX Security 2025.

## Requirements

Private Investigator requires a Linux machine with an NVIDIA graphics card.
Due to the large size of language models and data, we recommend a machine with at
least 32 CPU cores, 128 GB of system memory
24 GB of GPU memory, and 150 GB of disk space.
Our tests were conducted on a machine running Ubuntu 22.04 (64-bit) with NVIDIA GeForce RTX 3090 GPU.
We also provide the expected time required to run each experiments with one GPU that we used.


## Model Access Token

To use OpenELM in the experiments, generate a token of you Hugging Face account and gain access 
on the Llama-2-7b model (https://huggingface.co/meta-llama/Llama-2-7b). 
Type your token during executing command for launching Docker contatiner.
In case OpenELM would be not used, this procedure can be ignored.


## Installation

To run our scripts, the following software dependencies must be installed:

1. **Docker**

   Install Docker by following the instructions
   [here](https://docs.docker.com/engine/install/).

2. **Run Docker without root privileges**

   Our scripts are designed to avoid requiring root privileges. To manage Docker
   as a non-root user, follow the instructions
   [here](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

3. **CUDA Toolkit 12.1**

   Private Investigator is tested with CUDA Toolkit 12.1. Please install CUDA Toolkit 12.1 from
   this [link](https://developer.nvidia.com/cuda-12-1-0-download-archive). If
   using a different CUDA version, update the base image in the Dockerfile
   accordingly. Available base images can be found
   [here](https://hub.docker.com/r/pytorch/pytorch/tags).

4. **NVIDIA Container Toolkit**

   To enable GPU access within Docker containers, install the NVIDIA Container
   Toolkit by following the guide
   [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Usage - Main Experiments (Table 2)

To obtain main results in Table 2, execute Step 1 through Step 6 sequentially.


### Step 1: Launch the Docker container

Build the Docker image using the provided Dockerfile and start a Docker
container.

```
$ git clone --recurse-submodules https://github.com/WSP-LAB/PrivateInvestigator
$ cd scripts
$ ./build.sh [MODEL_ACCESS_TOKEN]
$ ./launch_container.sh
```

### Step 2: Download Datasets

**From step 2, instructions should be run inside the Docker container.**

Download Enron, TREC, and commoncrawl dataset.

```
$ cd scripts
$ ./get_enron.sh
$ ./get_trec.sh
$ ./get_commoncrawl.sh
```

### Step 3: Download fine-tuned model weights

Download all model weights required for the main experiments and countermeasures.

```
$ cd scripts
$ ./get_model_weights.sh
$ ./get_model_weights_counter.sh
```



### Step 4: Generate prompts (Phase I of Private Investigator)

Generating all prompts takes approximately 600 GPU hours.
Thus, we provide three options for this prompt generation experiments.

(1) Use pre-generated prompts (skip step 4).

We provide prompts that we used in our experiments in `experiments/prompts` folder.

(2) Generate prompts using interim results (less than an hour).

To shorten the time required to generate prompts, use interim results for prompt generation
by executing the `run_generate_prompt_interim.sh` script
We provide prompt candidates and corresponding number of PIIs generated using such prompt
by the surrogate model in `experiments/text_generation` folder.
These scripts will overwrite the provided prompts in `experiments/prompts`.

```
$ cd experiments
$ ./run_generate_prompt_interim.sh
```

(3) Generate prompts from scratch (about 600 GPU hours).

Generate prompt from the scratch by running the `run_generate_prompt_scratch.sh` script.
This will overwrite the provided results in `experiments/text_generation` and 
`experiments/prompts`.

```
$ cd experiments
$ ./run_generate_prompt_scratch.sh
```


### Step 5: Extract PIIs (Phase II of Private Investigator)

Extracting all three types of PIIs from all target models takes more than 200 GPU hours.
Thus, we provide two options for this PII extraction experiments.

(1) Extract email addresses from GPT-Neo (about 15 GPU hours).

Extract email addresses from GPT-Neo model using the generated prompts by executing `run_extract_piis_small.sh`.

```
$ cd experiments
$ ./run_extract_piis_small.sh
```

(2) Extract all three types of PIIs from all target models (about 200 GPU hours).

Extract email addresses, phone numbers, and person names by executing `run_extract_piis_all.sh`.

```
$ cd experiments
$ ./run_extract_piis_all.sh
```

PII extraction results will be stored in the `experiments/all_attack_result.txt` file.

### Step 6: Run baselines

Extract PIIs by 4 baselines: Carlini-topk, Carlini-temperature, Carlini-internet, and Lukas.

(1) Extract email addresses from GPT-Neo (about 16 GPU hours).

```
$ cd experiments
$ ./run_baselines_small.sh
```

(2) Extract all three types of PIIs from all target models (about 400 GPU hours).

```
$ cd experiments
$ ./run_baselines_email.sh
$ ./run_baselines_phone.sh
$ ./run_baselines_name.sh
```

PII extraction results will be stored in the `experiments/all_attack_result.txt` file.


## Usage - Analysis

Instructions below analyze the main experiment results obtained by preceding main experiments, 
so the main experiments above should be executed before running the below analysis experiments.
Each script reproduces the results in Section 5 and 6, 
and all experiments are denoted with the corresponding result table and figure in our paper.
Note that each analysis experiment could be executed independently.


### PII overlap (Figure 2, 7, 10, and 11)

Count the number of exclusive / common PIIs extracted by Private Investigator and baselines (Figure 2, 10, and 11).
Count the number of exclusive / common PIIs extracted by Private Investigator using pre-trained surrogate model
and fine-tuned surrogate model (Figure 7).

```
$ cd experiments
$ python analyze_pii_overlap.py
```

Results will be stored in the `experiments/overlap.txt` file.

### PII duplication (Figure 3)

Categorize the uniquely extracted PII items by the number of times the PII items appear in the training data
by executing `analyze_pii_duplicate.py`.

```
$ cd experiments
$ python analyze_pii_duplicate.py
```

Results will be stored in the `experiments/figure` directory.

### PII length (Figure 4)

Measure the impact of the token length on number of extracted email addresses
by running `analyze_pii_length.py`.
To analyze PII length, Differential Privacy experiments should be executed before.

```
$ cd experiments
$ python analyze_pii_length.py
```

Results will be stored in the `experiments/figure` directory.

### PII perplexity (Table 3)

Measure the perplexity of PIIs extracted by Private Investigator and baselines
by running `analyze_pii_perplexity.py`.

```
$ cd experiments
$ python analyze_pii_perplexity.py
```

Results will be stored in the `experiments/perplexity.txt` file.

### Contextual similarity (Figure 5 and Table 9)

Compute contextual similarity between the latent vectors of
preceding texts for PII items exclusively extracted by Private
Investigator, the corresponding latent vectors from the
training data, and the latent vectors of preceding texts
for PII items exclusively extracted by baseline attacks 
by running `run_analysis_contextual_similarity_small.sh` and 
`run_analysis_contextual_similarity_all.sh`.

(1) For email addresses from GPT-Neo (about 2 GPU hours).

```
$ cd experiments
$ ./run_analysis_contextual_similarity_small.sh
```

(2) For all three types of PIIs from all target models (about 150 GPU hours).

```
$ cd experiments
$ ./run_analysis_contextual_similarity_all.sh
```

Results will be stored in `experiments/contextual_similarity.txt` file.


### PII-eliciting Directions (Figure 8, 13, and 14)

Compute cosine similarity between
the oracle PII-eliciting directions and last-token latent vectors
of all single-token prompts or our prompts generated by
the surrogate model by running `run_analysis_pii_eliciting_direction.sh`.

```
$ cd experiments
$ ./run_analysis_pii_eliciting_direction.sh
```

Results will be stored in the `experiments/figure` directory.



### Deduplication (Table 6)

Execute Private Investigator and baselines on GPT-Neo trained with deduplicated dataset by executing 
`run_experiments_dedup.sh`. It takes about 100 GPU hours.

```
$ cd experiments
$ ./run_experiments_dedup.sh
```
The attack results will be saved in the `experiments/all_attack_result.txt` file.

### Differential Privacy (Table 7)

Execute Private Investigator and baselines on GPT-Neo trained with DP-SGD by executing 
`run_experiments_dp.sh`. It takes about 100 GPU hours.

```
$ cd experiments
$ ./run_experiments_dp.sh
```
The attack results will be saved in the `experiments/all_attack_result.txt` file.

