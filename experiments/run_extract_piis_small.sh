# extracting emails from GPT-Neo model
python run_attack_campaign.py --architecture gptneo --model_ckpt weights/gptneo-enron --surrogate pre --target email --c 0.25
python run_attack_campaign.py --architecture gptneo --model_ckpt weights/gptneo-enron --surrogate fine --target email --c 0.25
python run_attack_campaign.py --architecture gptneo --model_ckpt weights/gptneo-trec --surrogate pre --target email --c 0.25
python run_attack_campaign.py --architecture gptneo --model_ckpt weights/gptneo-trec --surrogate fine --target email --c 0.25
