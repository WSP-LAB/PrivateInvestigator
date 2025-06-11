
token="$1"

sed -i "s/Model_Access_Token/$token/g" ../src/pii_leakage/models/language_model.py
sed -i "s/Model_Access_Token/$token/g" ../src/carlini/extraction.py


docker build -t private_investigator ..
docker run -d --name private_investigator --gpus all -it private_investigator