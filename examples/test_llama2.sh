
python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="None" \
   --txt="data/context.6719.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="leaky_rerope" \
   --txt="data/context.6719.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="llama_scaling" \
   --alpha="auto" \
   --txt="data/context.6719.txt" \
   --repeat="3"


python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="None" \
   --txt="data/context.14592.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="leaky_rerope" \
   --txt="data/context.14592.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="llama_scaling" \
   --alpha="auto" \
   --txt="data/context.14592.txt" \
   --repeat="3"


python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="None" \
   --txt="data/context.10372.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="leaky_rerope" \
   --txt="data/context.10372.txt" \
   --repeat="3"

python llama2_test.py \
	--model_path="models/openbuddy-llama2-13b-v8.1-fp16" \
 	--temp="llama2" \
   --gpus="0,1,2,3" \
   --pe="llama_scaling" \
   --alpha="auto" \
   --txt="data/context.10372.txt" \
   --repeat="3"