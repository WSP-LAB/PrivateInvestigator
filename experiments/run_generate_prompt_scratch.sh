
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target email  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target phone  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --surrogate pre  --target name  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target email  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target phone  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target name  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target email  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target phone  --from_scratch True
python run_generate_prompt.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target name  --from_scratch True
