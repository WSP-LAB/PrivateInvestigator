## email
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method topk  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method topk  --target email

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method temperature  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method temperature  --target email

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method internet  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method internet  --target email

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_deduplicated  --target email
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_deduplicated  --target email


## phone
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method topk  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method topk  --target phone

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method temperature  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method temperature  --target phone

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method internet  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method internet  --target phone

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_deduplicated  --target phone
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_deduplicated  --target phone


## name
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method topk  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method topk  --temp 1.4  --target name

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method temperature  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method temperature  --temp 1.4  --target name

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_deduplicated  --method internet  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_deduplicated  --method internet  --temp 1.4  --target name

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_deduplicated  --temp 1.4  --target name
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_deduplicated  --temp 1.4  --target name
