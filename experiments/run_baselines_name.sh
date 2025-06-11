## extracting person names from GPT2
# carlini top-k
python baseline_carlini.py  --model_path weights/gpt2-enron  --method topk  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gpt2-trec  --method topk  --temp 1.4  --target name

# carlini temperature
python baseline_carlini.py  --model_path weights/gpt2-enron  --method temperature  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gpt2-trec  --method temperature  --temp 1.4  --target name

# carlini internet
python baseline_carlini.py  --model_path weights/gpt2-enron  --method internet  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gpt2-trec  --method internet  --temp 1.4  --target name

# lukas
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --temp 1.4  --target name
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --temp 1.4  --target name


## extracting person names from GPT-Neo
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron  --method topk  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec  --method topk  --temp 1.4  --target name

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron  --method temperature  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec  --method temperature  --temp 1.4  --target name

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron  --method internet  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec  --method internet  --temp 1.4  --target name

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --temp 1.4  --target name
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --temp 1.4  --target name


## extracting person names from OpenELM
# carlini top-k
python baseline_carlini.py  --model_path weights/openelm-enron  --method topk  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/openelm-trec  --method topk  --temp 1.4  --target name

# carlini temperature
python baseline_carlini.py  --model_path weights/openelm-enron  --method temperature  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/openelm-trec  --method temperature  --temp 1.4  --target name

# carlini internet
python baseline_carlini.py  --model_path weights/openelm-enron  --method internet  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/openelm-trec  --method internet  --temp 1.4  --target name

# lukas
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-enron  --temp 1.4  --target name
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-trec  --temp 1.4  --target name


## extracting person names from PHI2
# carlini top-k
python baseline_carlini.py  --model_path weights/phi2-enron  --method topk  --temp 1.4  --target name  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method topk  --temp 1.4  --target name  --batch_size 64

# carlini temperature
python baseline_carlini.py  --model_path weights/phi2-enron  --method temperature  --temp 1.4  --target name  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method temperature  --temp 1.4  --target name  --batch_size 64

# carlini internet
python baseline_carlini.py  --model_path weights/phi2-enron  --method internet  --temp 1.4  --target name  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method internet  --temp 1.4  --target name  --batch_size 64

# lukas
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-enron  --temp 1.4  --target name  --eval_batch_size 64
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-trec  --temp 1.4  --target name  --eval_batch_size 64



