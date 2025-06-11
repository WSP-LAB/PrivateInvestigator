
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target email
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target phone
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target name
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target email
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target phone
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target name
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target email
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target phone
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target name