# extract email addresses
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate fine  --target email  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate fine  --target email  --c 0.25

python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target email  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target email  --c 0.25

python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate fine  --target email  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate pre  --target email  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate fine  --target email  --c 0.25

python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate pre  --target email  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate fine  --target email  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate pre  --target email  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate fine  --target email  --c 0.25  --eval_batch_size 64

# extract phone numbers
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate fine  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate fine  --target phone  --c 0.25

python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target phone  --c 0.25

python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate fine  --target phone  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate pre  --target phone  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate fine  --target phone  --c 0.25

python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate pre  --target phone  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate fine  --target phone  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate pre  --target phone  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate fine  --target phone  --c 0.25  --eval_batch_size 64

# extract person names
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-enron  --surrogate fine  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gpt2  --model_ckpt weights/gpt2-trec  --surrogate fine  --target name  --temp 1.4  --c 0.25

python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-enron  --surrogate fine  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture gptneo  --model_ckpt weights/gptneo-trec  --surrogate fine  --target name  --temp 1.4  --c 0.25

python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-enron  --surrogate fine  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate pre  --target name  --temp 1.4  --c 0.25
python run_attack_campaign.py  --architecture openelm  --model_ckpt weights/openelm-trec  --surrogate fine  --target name  --temp 1.4  --c 0.25

python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate pre  --target name  --temp 1.4  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-enron  --surrogate fine  --target name  --temp 1.4  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate pre  --target name  --temp 1.4  --c 0.25  --eval_batch_size 64
python run_attack_campaign.py  --architecture phi2  --model_ckpt weights/phi2-trec  --surrogate fine  --target name  --temp 1.4  --c 0.25  --eval_batch_size 64
