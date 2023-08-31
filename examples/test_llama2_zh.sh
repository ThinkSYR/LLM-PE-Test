
python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
   --gpus="0,1,2,3" \
   --pe="None" \
   --txt="data/context.10372.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
   --gpus="0,1,2,3" \
   --pe="leaky_rerope" \
   --txt="data/context.14592.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/chinese-alpaca-2-7b-hf" \
 	--temp="llama2_zh" \
   --gpus="0,1,2,3" \
   --pe="llama_scaling" \
   --alpha="auto" \
   --txt="data/context.10372.txt" \
   --repeat="3"

