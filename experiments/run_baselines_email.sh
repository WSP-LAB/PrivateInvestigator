## extracting email addresses from GPT2
# carlini top-k
python baseline_carlini.py  --model_path weights/gpt2-enron  --method topk  --target email
python baseline_carlini.py  --model_path weights/gpt2-trec  --method topk  --target email

# carlini temperature
python baseline_carlini.py  --model_path weights/gpt2-enron  --method temperature  --target email
python baseline_carlini.py  --model_path weights/gpt2-trec  --method temperature  --target email

# carlini internet
python baseline_carlini.py  --model_path weights/gpt2-enron  --method internet  --target email
python baseline_carlini.py  --model_path weights/gpt2-trec  --method internet  --target email

# lukas
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --target email
python baseline_lukas.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --target email


## extracting email addresses from GPT-Neo
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron  --method topk  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec  --method topk  --target email

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron  --method temperature  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec  --method temperature  --target email

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron  --method internet  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec  --method internet  --target email

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --target email
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --target email


## extracting email addresses from OpenELM
# carlini top-k
python baseline_carlini.py  --model_path weights/openelm-enron  --method topk  --target email
python baseline_carlini.py  --model_path weights/openelm-trec  --method topk  --target email

# carlini temperature
python baseline_carlini.py  --model_path weights/openelm-enron  --method temperature  --target email
python baseline_carlini.py  --model_path weights/openelm-trec  --method temperature  --target email

# carlini internet
python baseline_carlini.py  --model_path weights/openelm-enron  --method internet  --target email
python baseline_carlini.py  --model_path weights/openelm-trec  --method internet  --target email

# lukas
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-enron  --target email
python baseline_lukas.py  --architecture openelm  --model_ckpt weights/openelm-trec  --target email


## extracting email addresses from PHI2
# carlini top-k
python baseline_carlini.py  --model_path weights/phi2-enron  --method topk  --target email  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method topk  --target email  --batch_size 64

# carlini temperature
python baseline_carlini.py  --model_path weights/phi2-enron  --method temperature  --target email  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method temperature  --target email  --batch_size 64

# carlini internet
python baseline_carlini.py  --model_path weights/phi2-enron  --method internet  --target email  --batch_size 64
python baseline_carlini.py  --model_path weights/phi2-trec  --method internet  --target email  --batch_size 64

# lukas
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-enron  --target email  --eval_batch_size 64
python baseline_lukas.py  --architecture phi2  --model_ckpt weights/phi2-trec  --target email  --eval_batch_size 64



