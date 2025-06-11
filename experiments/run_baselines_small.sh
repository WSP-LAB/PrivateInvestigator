# extracting emails from GPT-Neo model

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
