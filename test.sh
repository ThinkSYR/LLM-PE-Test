# input context length: 11321 54G
# input context length: 14836 64G
python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
    --gpus="0,1,2,3" \
    --pe="None"

# input context length: 14836 80G
python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
    --gpus="0,1,2,3" \
    --pe="rerope" \
    --txt="data/context.6719.txt" \
    --repeat="1"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
    --gpus="0,1,2,3" \
    --pe="rerope" \
    --txt="data/context.14592.txt" \
    --repeat="2"

# input context length: 14836 80G
python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
    --gpus="0,1,2,3" \
    --pe="leaky_rerope"