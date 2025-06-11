## email
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method topk  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method topk  --target email

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method temperature  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method temperature  --target email

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method internet  --target email
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method internet  --target email

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_dp  --target email
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_dp  --target email


## phone
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method topk  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method topk  --target phone

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method temperature  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method temperature  --target phone

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method internet  --target phone
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method internet  --target phone

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_dp  --target phone
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_dp  --target phone


## name
# carlini top-k
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method topk  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method topk  --temp 1.4  --target name

# carlini temperature
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method temperature  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method temperature  --temp 1.4  --target name

# carlini internet
python baseline_carlini.py  --model_path weights/gptneo-enron_dp  --method internet  --temp 1.4  --target name
python baseline_carlini.py  --model_path weights/gptneo-trec_dp  --method internet  --temp 1.4  --target name

# lukas
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-enron_dp  --temp 1.4  --target name
python baseline_lukas.py  --architecture gptneo  --model_ckpt weights/gptneo-trec_dp  --temp 1.4  --target name
