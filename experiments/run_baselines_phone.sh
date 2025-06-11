## extracting phone numbers from GPT2
# carlini top-k
python baseline_carlini.py  --model_path weights/gpt2-enron  --method topk  --target phone
python baseline_carlini.py  --model_path weights/gpt2-trec  --method topk  --target phone

# carlini temperature
python baseline_carlini.py  --model_path weights/gpt2-enron  --method temperature  --target phone
python baseline_carlini.py  --model_path weights/gpt2-trec  --method temperature  --target phone

# carlini internet
python baseline_carlini.py  --model_path weights/gpt2-enron  --method internet  --target phone
python baseline_carlini.py  --model_path weights/gpt2-trec  --method internet  --target phone

# lukas
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --target phone
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --target phone


## extracting phone numbers from GPT-Neo
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron  --method topk  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec  --method topk  --target phone

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron  --method temperature  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec  --method temperature  --target phone

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron  --method internet  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec  --method internet  --target phone

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --target phone
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --target phone


## extracting phone numbers from OpenELM
# carlini top-k
python baseline_carlini.py  --model_path weights/openelm-enron  --method topk  --target phone
python baseline_carlini.py  --model_path weights/openelm-trec  --method topk  --target phone

# carlini temperature
python baseline_carlini.py  --model_path weights/openelm-enron  --method temperature  --target phone
python baseline_carlini.py  --model_path weights/openelm-trec  --method temperature  --target phone

# carlini internet
python baseline_carlini.py  --model_path weights/openelm-enron  --method internet  --target phone
python baseline_carlini.py  --model_path weights/openelm-trec  --method internet  --target phone

# lukas
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-enron  --target phone
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-trec  --target phone


## extracting phone numbers from PHI2
# carlini top-k
python baseline_carlini.py  --model_path weights/phi2-enron  --method topk  --target phone  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method topk  --target phone  --batch_size 64

# carlini temperature
python baseline_carlini.py  --model_path weights/phi2-enron  --method temperature  --target phone  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method temperature  --target phone  --batch_size 64

# carlini internet
python baseline_carlini.py  --model_path weights/phi2-enron  --method internet  --target phone  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method internet  --target phone  --batch_size 64

# lukas
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-enron  --target phone  --eval_batch_size 64
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-trec  --target phone  --eval_batch_size 64



