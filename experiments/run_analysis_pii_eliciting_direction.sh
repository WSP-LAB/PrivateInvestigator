python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-enron --target email
python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-enron --target phone
python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-enron --target name

python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-trec --target email
python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-trec --target phone
python analyze_pii_eliciting_direction.py --architecture gptneo --model_ckpt weights/gptneo-trec --target name

